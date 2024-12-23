import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
import seaborn as sns  # For enhanced plotting

def vector_similarity(actual_changes, predicted_changes, mode='cosine'):
    """
    Compute a similarity metric (cosine or Pearson correlation)
    between two lists of price changes.
    """
    actual_arr = np.array(actual_changes)
    pred_arr = np.array(predicted_changes)
    
    if len(actual_arr) < 2:
        return np.nan

    if mode == 'pearson':
        # Pearson correlation: returns value in [-1, +1]
        return np.corrcoef(actual_arr, pred_arr)[0, 1]
    elif mode == 'cosine':
        # Cosine similarity: also in [-1, +1]
        dot = np.dot(actual_arr, pred_arr)
        norm_a = np.linalg.norm(actual_arr)
        norm_p = np.linalg.norm(pred_arr)
        if norm_a == 0 or norm_p == 0:
            return np.nan
        return dot / (norm_a * norm_p)
    else:
        raise ValueError("Unknown mode. Use 'pearson' or 'cosine'.")

def simulate_one_step(params, P_current, C1_prev, N1_prev, C2_prev, N2_prev,
                      per_1=1.0, per_2=-1.0, P_prev_prev=None):
    """
    Single-step price simulation with 2 groups, returning predicted next price & updated states.
    """
    alpha, beta, x = params[0], params[1], params[2]

    delta_P_prev_frac = (P_current - P_prev_prev) / P_prev_prev
    new_x = x + delta_P_prev_frac/2

    # Weighted "investment intensity"
    I1_prev = per_1 * delta_P_prev_frac * alpha
    I2_prev = per_2 * delta_P_prev_frac * alpha
    
    # Current "wealth" per group
    B1_prev = C1_prev + N1_prev * P_current
    B2_prev = C2_prev + N2_prev * P_current

    # Update holdings
    N1_new = N1_prev + I1_prev * B1_prev
    N2_new = N2_prev + I2_prev * B2_prev

    delta_N1 = N1_new - N1_prev
    delta_N2 = N2_new - N2_prev

    # Adjust cash using midpoint
    midpoint_price = (P_current + P_prev_prev) / 2
    C1_new = C1_prev - delta_N1 * midpoint_price
    C2_new = C2_prev - delta_N2 * midpoint_price

    # Next price update
    delta_P_n = beta * (
        new_x * (I1_prev * B1_prev) +
        (1 - new_x) * (I2_prev * B2_prev)
    )
    P_next_pred = P_current + delta_P_n
    return float(P_next_pred), C1_new, N1_new, C2_new, N2_new

def residuals(params, P_train, per_1=1.0, per_2=-1.0):
    """
    Residual function used by least_squares.
    """
    alpha, beta, x = params[0], params[1], params[2]
    C1_init, N1_init, C2_init, N2_init = params[3], params[4], params[5], params[6]

    # Initialize states
    C1 = C1_init
    N1 = N1_init
    C2 = C2_init
    N2 = N2_init

    residual_list = []
    predicted_values = []
    
    # Start from i=1 => ensures P_prev_prev is real data
    for i in range(1, len(P_train) - 1):
        P_prev_prev = P_train[i - 1]
        P_current = P_train[i]
        P_next_actual = P_train[i + 1]

        P_next_pred, C1, N1, C2, N2 = simulate_one_step(
            params, P_current, C1, N1, C2, N2,
            per_1=per_1, per_2=per_2,
            P_prev_prev=P_prev_prev
        )
        residual = P_next_pred - P_next_actual
        residual_list.append(residual)
        predicted_values.append(P_next_pred)

    # Penalty for large states
    penalty_scale_states = 1e-10
    state_penalty = penalty_scale_states * (C1**2 + N1**2 + C2**2 + N2**2)

    # Penalty for large predictions
    prediction_penalty_scale = 1e-8
    prediction_penalty = sum(prediction_penalty_scale * (p**2) for p in predicted_values)

    if residual_list:
        residual_list[-1] += state_penalty + prediction_penalty

    return np.array(residual_list)

def process_ticker(ticker, start_date, end_date, window_size, forecast_horizon, smooth_window):
    """
    For each ticker:
      1) Download data
      2) Fit smoothed & unsmoothed models
      3) Bet exactly ±1 share each day for both smoothed & unsmoothed
      4) Baseline: ±10 shares if yesterday up/down
      5) Buy & Hold: 10 shares from start to end
      6) Compute similarity in daily changes (rather than directional accuracy)
      7) Return results
    """
    try:
        data = yf.download(ticker, start=start_date, end=end_date)
        if data.empty:
            return {"ticker": ticker, "status": "no_data_returned"}
    except ValueError as e:
        return {"ticker": ticker, "status": f"download_error: {e}"}
    
    # Identify if 'Adj Close' or 'Close' is present
    if 'Adj Close' in data.columns:
        P_obs_full = data['Adj Close'].values
    elif 'Close' in data.columns:
        P_obs_full = data['Close'].values
    else:
        return {"ticker": ticker, "status": "no_valid_price_column"}

    P_obs_full = P_obs_full.flatten()
    if len(P_obs_full) == 0:
        return {"ticker": ticker, "status": "no_data"}

    # Create smoothed data
    if len(P_obs_full) < smooth_window:
        return {"ticker": ticker, "status": "not_enough_data_for_smoothing"}
    P_smooth = np.convolve(P_obs_full, np.ones(smooth_window)/smooth_window, mode='valid')
    offset = smooth_window - 1
    P_obs_aligned = P_obs_full[offset:]

    if len(P_obs_aligned) <= window_size:
        return {"ticker": ticker, "status": "not_enough_data_after_alignment"}

    # Buy & Hold
    buy_and_hold_shares = 10.0/P_obs_aligned[0]
    buy_and_hold_initial_price = P_obs_aligned[0]
    buy_and_hold_final_price = P_obs_aligned[-1]
    buy_and_hold_profits = buy_and_hold_shares * (buy_and_hold_final_price - buy_and_hold_initial_price)

    # Prepare for smoothed & unsmoothed models
    initial_guess = [1e-5, 1e-5, 0.5, 1000.0, 1.0, 1000.0, 1.0]
    lb = [1e-7, 0,  0.0, -1e4, -1e3, -1e4, -1e3]
    ub = [1e-4, 1e-4, 1.0, 1e4, 1e3, 1e4, 1e3]

    predictions_smooth = []
    actuals_smooth = []
    profits_smooth = 0.0

    predictions_unsmoothed = []
    actuals_unsmoothed = []
    profits_unsmoothed = 0.0

    # We'll track daily changes for similarity:
    smooth_actual_changes = []
    smooth_predicted_changes = []
    unsmooth_actual_changes = []
    unsmooth_predicted_changes = []

    # Baseline
    baseline_shares = 10.0/P_obs_aligned[0]
    baseline_profits = 0.0

    guess_smooth = initial_guess.copy()
    guess_unsmoothed = initial_guess.copy()

    # Walk-forward
    for current_day in range(window_size, len(P_smooth) - forecast_horizon, forecast_horizon):
        P_train_smooth = P_smooth[current_day - window_size : current_day + 1]
        P_train_unsmoothed = P_obs_aligned[current_day - window_size : current_day + 1]

        if len(P_train_smooth) < window_size+1 or len(P_train_unsmoothed) < window_size+1:
            break

        # Fit smoothed
        res_smooth = least_squares(
            residuals, guess_smooth,
            args=(P_train_smooth,),
            bounds=(lb, ub),
            method='trf',
            ftol=1e-8,
            xtol=1e-8,
            verbose=0
        )
        fitted_smooth = res_smooth.x

        # Fit unsmoothed
        res_unsmoothed = least_squares(
            residuals, guess_unsmoothed,
            args=(P_train_unsmoothed,),
            bounds=(lb, ub),
            method='trf',
            ftol=1e-8,
            xtol=1e-8,
            verbose=0
        )
        fitted_unsmoothed = res_unsmoothed.x

        if current_day + forecast_horizon >= len(P_obs_aligned):
            break
        actual_next = P_obs_aligned[current_day + forecast_horizon]
        current_original = P_obs_aligned[current_day]

        # SMOOTHED Prediction
        P_current_s = P_train_smooth[-1]
        P_prev_prev_s = P_train_smooth[-2]
        P_next_pred_s, *_ = simulate_one_step(
            fitted_smooth, P_current_s, 1000.0, 1.0, 1000.0, 1.0,
            P_prev_prev=P_prev_prev_s
        )
        predictions_smooth.append(P_next_pred_s)
        actuals_smooth.append(actual_next)

        # Profit calculation
        predicted_dir_s = np.sign(P_next_pred_s - current_original)
        bet_shares_s = predicted_dir_s  
        profit_s = bet_shares_s * (actual_next - current_original)
        profits_smooth += profit_s

        # NEW: store daily changes for similarity
        smooth_actual_changes.append(actual_next - current_original)
        smooth_predicted_changes.append(P_next_pred_s - current_original)

        # UNSMOOTHED Prediction
        P_current_u = P_train_unsmoothed[-1]
        P_prev_prev_u = P_train_unsmoothed[-2]
        P_next_pred_u, *_ = simulate_one_step(
            fitted_unsmoothed, P_current_u, 1000.0, 1.0, 1000.0, 1.0,
            P_prev_prev=P_prev_prev_u
        )
        predictions_unsmoothed.append(P_next_pred_u)
        actuals_unsmoothed.append(actual_next)

        predicted_dir_u = np.sign(P_next_pred_u - current_original)
        profit_u = predicted_dir_u * (actual_next - current_original)
        profits_unsmoothed += profit_u

        # NEW: store daily changes for similarity
        unsmooth_actual_changes.append(actual_next - current_original)
        unsmooth_predicted_changes.append(P_next_pred_u - current_original)

        # BASELINE: if price up yesterday => +10, else -10
        if current_day > 0:
            if P_obs_aligned[current_day] > P_obs_aligned[current_day - 1]:
                baseline_bet_shares = baseline_shares
            else:
                baseline_bet_shares = -baseline_shares
        else:
            baseline_bet_shares = baseline_shares
        baseline_profits += baseline_bet_shares * (actual_next - current_original)

        # Update guesses
        guess_smooth = fitted_smooth
        guess_unsmoothed = fitted_unsmoothed

    # Compute final similarity
    similarity_smooth = vector_similarity(smooth_actual_changes, smooth_predicted_changes, mode='cosine')
    similarity_unsmoothed = vector_similarity(unsmooth_actual_changes, unsmooth_predicted_changes, mode='cosine')

    return {
        "ticker": ticker,
        # Instead of direction accuracy, we return similarity
        "similarity_smooth": similarity_smooth,
        "similarity_unsmoothed": similarity_unsmoothed,

        # We keep profit metrics & buy/hold as before
        "profits_smooth": profits_smooth,
        "profits_unsmoothed": profits_unsmoothed,
        "baseline_profits": baseline_profits,
        "buy_and_hold_profits": buy_and_hold_profits*10
    }

def main():
    # Example list of tickers
    tickers = ["EMN", "ETN", "EBAY", "ECL", "EIX", "EW", "EA", "ELV",
        "EMR", "ENPH", "ETR", "EOG", "EPAM", "EQT", "EFX", "EQIX",
        "EQR", "ERIE", "ESS", "EL", "EG", "EVRG", "ES", "EXC",
        "EXPE", "EXPD", "EXR", "XOM", "FFIV", "FDS", "FICO", "FAST",
        "FRT", "FDX", "FIS", "FITB", "FSLR", "FE", "FI", "FMC",
        "F", "FTNT", "FTV", "FOXA", "FOX", "BEN", "FCX", "GRMN",
        "IT", "GE", "GEHC", "GEV", "GEN", "GNRC", "GD", "GIS",
        "GM", "GPC", "GILD", "GPN", "GL", "GDDY", "GS", "HAL",
        "HIG", "HAS", "HCA", "DOC", "HSIC", "HSY", "HES", "HPE",
        "HLT", "HOLX", "HD", "HON", "HRL", "HST", "HWM", "HPQ",
        "HUBB", "HUM", "HBAN", "HII", "IBM", "IEX", "IDXX", "ITW",
        "INCY", "IR", "PODD", "INTC", "ICE", "IFF", "IP", "IPG",
        "INTU", "ISRG", "IVZ", "INVH", "IQV", "IRM", "JBHT", "JBL",
        "JKHY", "J", "JNJ", "JCI", "JPM", "JNPR", "K", "KVUE",
        "KDP", "KEY", "KEYS", "KMB", "KIM", "KMI", "KKR", "KLAC",
        "KHC", "KR", "LHX", "LH", "LRCX", "LW", "LVS", "LDOS",
        "LEN", "LLY", "LIN", "LYV", "LKQ", "LMT", "L", "LOW",
        "LULU", "LYB", "MTB", "MRO", "MPC", "MKTX", "MAR", "MMC",
        "MLM", "MAS", "MA", "MTCH", "MKC", "MCD", "MCK", "MDT",
        "MRK", "META", "MET", "MTD", "MGM", "MCHP", "MU", "MSFT",
        "MAA", "MRNA", "MHK", "MOH", "TAP", "MDLZ", "MPWR", "MNST",
        "MCO", "MS"]
    print(len(tickers))
    start_date = "2023-01-01"
    end_date = "2024-12-20"
    window_size = 8
    forecast_horizon = 1
    smooth_window = 3

    df_rows = []

    for ticker in tickers:
        result = process_ticker(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            window_size=window_size,
            forecast_horizon=forecast_horizon,
            smooth_window=smooth_window
        )
        if "status" in result:
            print(f"Ticker: {ticker}, Status: {result['status']}")
            continue

        # Print summary
        sim_smooth = result["similarity_smooth"]
        sim_unsmooth = result["similarity_unsmoothed"]
        print(f"Ticker: {ticker}")
        print(f"  Similarity (Smoothed)   = {sim_smooth:.4f}")
        print(f"  Similarity (Unsmoothed) = {sim_unsmooth:.4f}")
        print(f"  Profit (Smoothed)       = {result['profits_smooth']:.2f}")
        print(f"  Profit (Unsmoothed)     = {result['profits_unsmoothed']:.2f}")
        print(f"  Baseline Profit         = {result['baseline_profits']:.2f}")
        print(f"  Buy & Hold Profit       = {result['buy_and_hold_profits']:.2f}\n")

        df_rows.append({
            'Ticker': ticker,
            'Similarity_Smoothed': result["similarity_smooth"],
            'Similarity_Unsmoothed': result["similarity_unsmoothed"],
            'Smoothed_Profit': result["profits_smooth"],
            'Unsmoothed_Profit': result["profits_unsmoothed"],
            'Baseline_Profit': result["baseline_profits"],
            'Buy_and_Hold_Profit': result["buy_and_hold_profits"]
        })

    # Summarize in a DataFrame
    df = pd.DataFrame(df_rows)
    if df.empty:
        print("No tickers processed successfully.")
        return

    print("\n=== Summary DataFrame ===")
    print(df)

    # (1) Compute correlation: Smoothed_Profit vs. Buy_and_Hold_Profit
    #                          Unsmoothed_Profit vs. Buy_and_Hold_Profit
    corr_smooth = df[['Smoothed_Profit', 'Buy_and_Hold_Profit']].corr().iloc[0, 1]
    corr_unsmoothed = df[['Unsmoothed_Profit', 'Buy_and_Hold_Profit']].corr().iloc[0, 1]

    print(f"\nCorrelation (Smoothed vs. Buy & Hold): {corr_smooth:.4f}")
    print(f"Correlation (Unsmoothed vs. Buy & Hold): {corr_unsmoothed:.4f}")

    # (2) Plot the average profit
    avg_smooth = df['Smoothed_Profit'].mean()
    avg_unsmoothed = df['Unsmoothed_Profit'].mean()
    avg_baseline = df['Baseline_Profit'].mean()
    avg_buyhold = df['Buy_and_Hold_Profit'].mean()

    avg_df = pd.DataFrame({
        'Strategy': ['Smoothed', 'Unsmoothed', 'Baseline', 'Buy & Hold'],
        'Average_Profit': [avg_smooth, avg_unsmoothed, avg_baseline, avg_buyhold]
    })

    sns.set_style("whitegrid")
    plt.figure(figsize=(8, 5))
    sns.barplot(data=avg_df, x='Strategy', y='Average_Profit', palette='Blues_d')
    plt.title('Average Profit Across Tickers')
    plt.xlabel('')
    plt.tight_layout()
    plt.show()

    # (3) Correlation matrix among the 4 profit columns
    profit_cols = ['Smoothed_Profit', 'Unsmoothed_Profit', 'Baseline_Profit', 'Buy_and_Hold_Profit']
    corr_matrix = df[profit_cols].corr()
    print("\n=== CORRELATION MATRIX (4 Profit Columns) ===")
    print(corr_matrix)

    # (4) Scatter map (pairplot) among the 4 profit columns
    plt.figure(figsize=(8, 6))
    sns.pairplot(df[profit_cols], corner=True, diag_kind='kde')
    plt.suptitle('Pairwise Scatter Plots (Profit Columns)', y=1.02)
    plt.show()

    # (5) NEW: Print tickers where Smoothed > Buy & Hold, and Unsmoothed > Buy & Hold
    smoothed_better = df[df['Smoothed_Profit'] > df['Buy_and_Hold_Profit']]['Ticker'].tolist()
    unsmoothed_better = df[df['Unsmoothed_Profit'] > df['Buy_and_Hold_Profit']]['Ticker'].tolist()

    print("\n=== Tickers where SMOOTHED > BUY & HOLD ===")
    print(smoothed_better)

    print("\n=== Tickers where UNSMOOTHED > BUY & HOLD ===")
    print(unsmoothed_better)

    print("\nDone!")

if __name__ == "__main__":
    main()
