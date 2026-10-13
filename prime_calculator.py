import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import lambertw
import time
import math


### function definitions ###
def is_palindrome(x:int) -> bool:
    """
    Determine if input parameter is a palindrome

    Args:
        x (int): input number

    Returns:
        bool: True if input is a palindrome
    """
    strx = str(x)
    return strx == strx[::-1]

def is_palindrome_vec(arr:list[int]) -> list[bool]:
    """
    Determine if array elements are palindromes

    Args:
        arr (list[int]): input numbers

    Returns:
        list[bool]: True if input array element is a palindrome
    """
    return [str(x)==str(x)[::-1] for x in arr]

def is_factor(x:int,y:int) -> bool:
    """
    Determine if x is a multiple of y

    Args:
        x,y (int): input numbers

    Returns:
        bool: True if x is a multiple of y
    """
    if int(y) == 0:
        return False
    if np.mod(int(x),int(y)) == 0:
        return True
    else:
        return False

def linear_sieve(nmax:int, filter_palindromes:bool=False) -> list[int]:
    """
    Use sieve approach to populate list of primes,
    starting at 2 and ending at user-defined limit

    Args:
        nmax (int): max integer to consider as prime
        filter_palindromes (bool): only retain primes that are palindromes

    Returns:
        list[int]: list of primes and their indices
    """
    prime = []

    # Create a boolean list "mark[0..n-1]" and  
    # initialize all entries of it as True. 
    # A value in mark[p] will finally be False 
    # if 'p' is Not a prime, else True. 
    mark = [True for i in range(nmax + 1)]
    p = 2
    while (p * p <= nmax):
        
        # If p is not changed, then it is a prime 
        if (mark[p] == True): 
            
            # Update all multiples of p 
            for i in range(p * p, nmax + 1, p): 
                mark[i] = False  
        p += 1
        
    # Retain prime numbers 
    for p in range(2, nmax + 1): 
        if mark[p]:
            prime.append(p)

    # Apply palindrome filter
    if filter_palindromes:
        primes_mask = is_palindrome_vec(prime)
        primes_filtered = [prime[i] for i in range(len(prime)) if primes_mask[i]]
    else:
        primes_filtered = prime

    return primes_filtered


def segmented_sieve(nmax:int, filter_palindromes:bool=False) -> list[int]:
    """
    Use segmented sieve approach to populate list of primes,
    starting at 2 and ending at user-defined limit.
    Can access higher primes while meeting memory constraints.

    Args:
        nmax (int): max integer to consider as prime
        filter_palindromes (bool): only retain primes that are palindromes

    Returns:
        list[int]: list of primes and their indices
    """

    if nmax < 2:
        return []

    # Begin with simple sieve
    limit = int(math.floor(math.sqrt(nmax)) + 1)
    base_primes = linear_sieve(limit)
    prime = list(base_primes)
    
    # Subdivide the range with segment size sqrt(n)
    low = limit + 1
    high = low + limit
    
    while low <= nmax:
        if high > nmax:
            high = nmax + 1
            
        mark = [True for i in range(limit + 1)]
        
        for i in range(len(base_primes)):
            
            # Find the smallest prime factor within [low ... high] 
            loLim = (low // base_primes[i]) * base_primes[i]
            if loLim < low:
                loLim += base_primes[i]
                
            # Mark multiples of base_primes[i] within [low ... high] 
            for j in range(loLim, high, base_primes[i]):
                mark[j - low] = False
                
        # Retain the primes
        for i in range(low, high):
            if mark[i - low]:
                prime.append(i)
                
        # Update low and high for next segment 
        low = low + limit
        high = high + limit

    # Apply palindrome filter
    if filter_palindromes:
        primes_mask = is_palindrome_vec(prime)
        primes_filtered = [prime[i] for i in range(len(prime)) if primes_mask[i]]
    else:
        primes_filtered = prime

    return primes_filtered

def _write_primes(nmax:int, output_file:str, filter_palindromes:bool) -> None:
    start = time.time()

    primes = segmented_sieve(nmax=nmax,filter_palindromes=filter_palindromes)

    df = pd.DataFrame({'index': range(1,len(primes)+1), 'n': primes})
    df['index']=df['index'].astype("UInt64")
    df['n']=df['n'].astype("UInt64")

    if output_file is not None:
        df.to_csv(output_file, index=False)
    else:
        print(df['n'].values)

    end = time.time()
    print(f"Operation took {end-start} seconds.")

def generate_primes(nmax:int, output_file:str=None) -> None:
    """
    Generates prime numbers from 0 to user-specified maximum integer.
    Primes are written to a CSV output file or console along with their index, 
    e.g. the prime 2 has index 1 (it's the 1st prime) and the prime 5 has
    index 3 (it's the 3rd prime).

    If output file is None, function writes primes to console (without their index)

    Args:
        nmax [int]: max integer to consider as prime
        output_file [str]: file path for output

    Returns:
        None
    """
    _write_primes(nmax, output_file, filter_palindromes=False)
    return None

def generate_prime_palindromes(nmax:int, output_file:str=None) -> None:
    """
    Generates the prime palindromes from 0 to user-specified maximum integer.
    Primes are written to a CSV output file or console along with their index, 
    e.g. the prime 2 has index 1 (it's the 1st prime) and the prime 5 has
    index 3 (it's the 3rd prime).

    If output file is None, function writes primes to console (without their index)

    Args:
        nmax [int]: max integer to consider as prime
        output_file [str]: file path for output

    Returns:
        None
    """
    _write_primes(nmax, output_file, filter_palindromes=True)
    return None


def cumulative_total_str_counts(df_series, substring:str):
    """
    Calculate cumulative totals of a given substring (integer or integers)
    within a pandas series.

    Args:
        df_series [pandas series]: series to search within
        substring [str]: pattern to search for (an integer or integers)

    Returns:
        counts [pandas series]: series of cumulative counts 
    """
    counts = df_series.str.count(substring).cumsum()
    return counts

def tabulate_total_str_chars(df_series):
    """
    Count the total number of characters in a pandas series.
    
    Args:
        df_series [pandas series]: series to inspect

    Returns: the total number of characters in the series 2nd column
    """
    temp = np.asarray(df_series).astype(str)
    lengths = np.char.str_len(temp)
    total_chars = np.sum(lengths)
    return total_chars

def tabulate_total_str_counts(df_series, substring:str):
    """
    Count the total number of occurrences of a given substring 
    (integer or integers) within a pandas series.

    Args:
        df_series [pandas series]: series to search within
        substring [str]: pattern to search for (an integer or integers)

    Returns:
        counts [int]: number of total counts of substring in df_series
    """
    counts = df_series.str.count(substring).sum()
    return counts

def tabulate_ending_str_counts(df_series, substring:str):
    """
    Count the total number of occurrences where the given pandas series
    ends with the given substring (integer or integers).

    Args:
        df_series [pandas series]: series to search within
        substring [str]: pattern to search for (an integer or integers)

    Returns:
        counts [int]: number of total counts of df_series ending in substring
    """
    counts = df_series.str.endswith(substring).sum()
    return counts

def compute_primes_stats(filepath:str):
    primes_listing = pd.read_csv(filepath)
    counts = 0
    total_counts = []
    ends_counts = []
    str_primes = primes_listing.n.astype(str)
    for i in range(0,10):
        stri = str(i)
        counts = tabulate_total_str_counts(str_primes, stri)
        total_counts.append(counts)
        counts = tabulate_ending_str_counts(str_primes, stri)
        ends_counts.append(counts)

    return [total_counts, ends_counts]
    
def compute_oom_stats(filepath:str, independent_bins:bool=False, relative_score:bool=False, as_fraction:bool=False):
    """
    Calculate occurrence rates for each digit within the specified dataset 
    (primes or prime palindromes) for each order of magnitude in the data.
    
    Args:
        filepath [filepath]: file containing dataset of interest (eg primes). Data
          must have index as first row and data (eg primes) as second row.
        independent_bins [bool]: should each OOM bin exclude all lower OOM integers. 
          For example, if True, OOM_2 would only include digits 10 through 99. 
        relative_score [bool]: should final data be shown as surplus/deficit counts relative
          to baseline expectation (10% of total number of counts for the OOM)
        as_fraction [bool]: divide final data by the total number of digits in the OOM,
          so that each OOM is on the same scale

    Returns: 
        df [Pandas dataframe]: row index is integer, columns are the orders of magnitude   
    """
    df_input = pd.read_csv(filepath)

    # count number of OOM in data
    max_oom = int(np.ceil(np.log10(df_input.loc[len(df_input)-1,'n'])))
    cols = ["OOM_"+str(i) for i in range(1,max_oom+1)]

    # N_digits (total number of occurrences of each digit in each OOM)
    df_digits = pd.DataFrame(columns=cols)
    df = pd.DataFrame(columns=cols)

    # compute relative fractions
    for i in range(1,max_oom+1):
        stri = str(i)
        if independent_bins:
            df_oom = df_input[(df_input.n >= (10**(i-1))) & (df_input.n < (10**i))]
        else:
            df_oom = df_input[df_input.n < (10**i)]
        df_oom_values = df_oom['n'].values
        N_digits = tabulate_total_str_chars(df_oom_values)

        for j in range(0,10):
            strj = str(j)
            countsj = tabulate_total_str_counts(df_oom.n.astype(str), strj)
            value = countsj
            if relative_score:
                value = value - 0.1*N_digits
            if as_fraction:
                value = value / N_digits if N_digits > 0 else np.nan
            df_digits.loc[j,"OOM_"+stri] = value

    return df_digits


def plot_magnitude_baselines(df):
    """
    data_matrix: 2D array (8 rows for magnitudes, 10 columns for integers 0-9).
    Values represent the surplus or deficit.
    """
    # 1. Transpose the compute_oom_stats output: rows become magnitudes
    # (OOM_1 first), columns become digits 0-9
    matrix_df = df.T
    data_matrix = matrix_df.to_numpy(dtype=float)
    n_mag = data_matrix.shape[0]

    magnitudes = [f'$10^{{{c.split("_")[1]}}}$' for c in matrix_df.index]
    x_values = np.arange(10)
    
    fig, ax = plt.subplots(figsize=(12, 1.2 * n_mag + 1))
    
    # We set each magnitude at a y-coordinate (1 through n_mag)
    # These serve as the "baselines"
    y_baselines = np.arange(1, n_mag + 1)

    # Define how 'tall' a bar can be before it hits the next category
    # 0.4 allows space for surplus and deficit without overlapping next baseline
    max_abs = np.nanmax(np.abs(data_matrix)) if np.any(np.isfinite(data_matrix)) else 0
    scaling_factor = 0.4 / (max_abs if max_abs != 0 else 1)

    for i, baseline in enumerate(y_baselines):
        # Draw the baseline for this magnitude
        ax.axhline(baseline, color='gray', linewidth=0.8, alpha=0.5)
        
        # Calculate bar heights relative to this specific baseline
        # surplus/deficit * scaling_factor ensures they stay within their lane
        heights = data_matrix[i, :] * scaling_factor
        
        # Color code: Green for surplus, Red for deficit
        colors = ['#2ecc71' if h > 0 else '#e74c3c' for h in heights]
        
        # Plot the bars starting FROM the baseline
        ax.bar(x_values, heights, bottom=baseline, color=colors, width=0.6, edgecolor='white', linewidth=0.5)

    # Formatting
    ax.set_yticks(y_baselines)
    ax.set_yticklabels(magnitudes)
    ax.set_xticks(x_values)
    ax.set_xlabel('Integer Categories (0-9)', fontsize=12)
    ax.set_ylabel('Order of Magnitude (Baseline)', fontsize=12)
    ax.set_title('Population Surplus/Deficit Relative to Magnitude Baselines', fontsize=14)
    
    # Remove top/right spines for a cleaner look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.show()

def newton_method(f, x0:float, df=None, tol:float=1e-10, max_iter:int=100) -> float:
    """
    Use Newton's method to find a zero of a function, starting from an initial guess.

    Args:
        f [callable]: function of one variable whose zero is sought
        x0 [float]: initial guess
        df [callable]: derivative of f. If None, a central-difference
          approximation is used
        tol [float]: stop when |f(x)| or the step size falls below this value
        max_iter [int]: maximum number of iterations

    Returns:
        x [float]: approximate zero of f
    """
    if df is None:
        def df(x):
            h = 1e-6 * max(1.0, abs(x))
            return (f(x + h) - f(x - h)) / (2 * h)

    x = float(x0)
    for i in range(max_iter):
        fx = f(x)
        if abs(fx) < tol:
            return x

        dfx = df(x)
        if dfx == 0:
            raise ZeroDivisionError(f"Derivative is zero at x = {x}; Newton's method cannot continue.")

        step = fx / dfx
        x = x - step
        if abs(step) < tol:
            return x

    raise RuntimeError(f"Newton's method did not converge within {max_iter} iterations (last x = {x}).")

def newton_prime_zeros(primes:list[int], tol:float=1e-12, max_iter:int=100) -> pd.DataFrame:
    """
    For each prime p, use Newton's method to find the zero of
    f(x) = ln(x) - ln(p)/x, i.e. the x that satisfies x^x = p.

    Args:
        primes [list[int]]: primes to iterate over (eg output of segmented_sieve)
        tol [float]: tolerance passed to newton_method
        max_iter [int]: maximum iterations passed to newton_method

    Returns:
        df [Pandas dataframe]: columns 'p' (the prime) and 'x' (the zero)
    """
    primes = list(primes)
    zeros = []
    for p in primes:
        lnp = math.log(p)
        f = lambda x: math.log(x) - lnp/x
        df = lambda x: 1/x + lnp/x**2

        # f is increasing and concave for x > 0, so starting left of the
        # zero (f(1) = -ln(p) < 0) converges from below without leaving x > 0
        zeros.append(newton_method(f, 1.0, df=df, tol=tol, max_iter=max_iter))

    return pd.DataFrame({'p': primes, 'x': zeros})

def lambert_w(y:float, k:int=0) -> tuple:
    """
    Compute branch k of the Lambert W function on a real argument y,
    i.e. a w that satisfies w * e^w = y. The default k = 0 is the principal
    branch W0, which is real for y >= -1/e.

    Args:
        y [float]: real argument (not nan). Must be >= -1/e when k = 0
        k [int]: branch of the Lambert W function

    Returns:
        (w, exp_w, y, ln_y) [tuple]: W_k(y), exp(W_k(y)), the argument y, and ln(y).
          w and exp_w are floats when k = 0 and complex otherwise.
          exp_w equals y / W_k(y) for y != 0, so lambert_w(ln(y), k) gives an x that
          satisfies x*ln(x) = ln(y), taking ln(x) = W_k(ln(y)).
          ln_y is -inf when y is 0 and nan when y is negative.
    """
    y = float(y)
    k = int(k)

    if math.isnan(y):
        raise ValueError("lambert_w is undefined for y = nan (eg ln of a negative number).")

    if k == 0:
        if y < -math.exp(-1):
            raise ValueError(f"W0 is not real for y < -1/e (got y = {y}).")

        w = float(lambertw(y, k=0).real)

        # The float -1/e rounds just below the true branch point, where scipy returns nan
        if math.isnan(w):
            w = -1.0

        exp_w = math.exp(w)
    else:
        w = complex(lambertw(y, k=k))
        exp_w = complex(np.exp(w))

    with np.errstate(divide='ignore', invalid='ignore'):
        ln_y = float(np.log(y))

    return w, exp_w, y, ln_y

def complex_prime_zeros(primes:list[int], k_values=range(0, 5)) -> pd.DataFrame:
    """
    For each prime p and each Lambert W branch k, compute
    x_k = exp(W_k(ln(p))), a solution of x*ln(x) = ln(p) with ln(x) = W_k(ln(p)).
    k = 0 gives the real solution, and all other k give complex solutions,
    and branches k and -k give complex conjugate solutions.

    Args:
        primes [list[int]]: primes to iterate over (eg output of segmented_sieve)
        k_values [iterable of int]: Lambert W branches to evaluate

    Returns:
        df [Pandas dataframe]: columns 'p', 'k', 'w' (W_k(ln(p)), i.e. ln(x_k))
          and 'x' (x_k), sorted by p and then k
    """
    primes = np.asarray(list(primes), dtype=np.int64)
    ln_p = np.log(primes)

    frames = []
    for k in k_values:
        w = lambertw(ln_p, k=int(k))
        frames.append(pd.DataFrame({'p': primes, 'k': int(k), 'w': w, 'x': np.exp(w)}))

    df = pd.concat(frames, ignore_index=True)
    return df.sort_values(['p', 'k'], kind='stable').reset_index(drop=True)

def lambert_w_primes(primes:list[int], k:int=0) -> pd.DataFrame:
    """
    For each prime p, call lambert_w(ln(p), k). The resulting w satisfies
    w * e^w = ln(p), so x = exp(w) satisfies x*ln(x) = ln(p), i.e. x^x = p.

    Args:
        primes [list[int]]: primes to iterate over (eg output of segmented_sieve)
        k [int]: Lambert W branch passed to lambert_w()

    Returns:
        df [Pandas dataframe]: one row per prime, with columns
          'p' (the prime), 'ln_p' (ln(p), the argument passed to lambert_w),
          'w(ln(p))' (W_k(ln(p)), i.e. ln(x)) and
          'exp_w(ln(p))' (exp(W_k(ln(p))), i.e. x).
          The last two are floats when k = 0 and complex otherwise.
    """
    rows = []
    for p in primes:
        w, exp_w, ln_p, _ = lambert_w(np.log(p), k=k)
        rows.append((p, ln_p, w, exp_w))
    return pd.DataFrame(rows, columns=['p', 'ln_p', 'w(ln(p))', 'exp_w(ln(p))'])

