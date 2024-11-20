def check_k(k, val):
    
    for i in range(len(k)):
        if k[i] % val == 0:
            print(f"Warning! k = {k[i]} is divisible for {val}")
