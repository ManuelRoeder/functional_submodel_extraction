import itertools
import math
import random


# omega_i(r) of Sec. 3.1 (n here, client weights p_n as q)
def coverage(C, rates, q):
    ks = [max(1, math.ceil(r * C)) for r in rates]
    n = []
    for k in range(1, C + 1):
        n.append(sum(qj for kj, qj in zip(ks, q) if kj >= k))
    return n, ks


# Eq. (4), left side: average over clients of sum_{k not in S} c_{i,k}
def pop_bound(c, perm, ks, q):
    tot = 0.0
    for kj, qj in zip(ks, q):
        tot += qj * sum(c[perm[k]] for k in range(kj, len(c)))
    return tot


# Eq. (4), right side
def pop_bound_via_coverage(c, perm, n):
    return sum((1.0 - n[k]) * c[perm[k]] for k in range(len(c)))


def main():
    random.seed(0)
    # Claim 2: Eq. (4); Claim 3: Prop. 1
    print("Claim 2 (identity) and Claim 3 (sorting optimal), brute force over all")
    print("permutations, many random capacity distributions:\n")

    worst_id_err = 0.0
    sorted_optimal_everywhere = True
    for trial in range(400):
        C = random.choice([5, 6, 7])
        c = [round(random.uniform(0.05, 3.0), 3) for _ in range(C)]
        m = random.choice([2, 3, 4])
        rates = sorted(random.sample([1.0, 0.5, 0.25, 0.125, 0.0625], m))
        w = [random.random() + 0.05 for _ in range(m)]
        q = [x / sum(w) for x in w]
        n, ks = coverage(C, rates, q)

        best, best_perm = None, None
        for perm in itertools.permutations(range(C)):
            direct = pop_bound(c, perm, ks, q)
            viacov = pop_bound_via_coverage(c, perm, n)
            worst_id_err = max(worst_id_err, abs(direct - viacov))
            if best is None or direct < best - 1e-12:
                best, best_perm = direct, perm

        # FedHOPE order (Algorithm 1, line 3); Prop. 1
        order = sorted(range(C), key=lambda i: -c[i])
        sorted_val = pop_bound(c, tuple(order), ks, q)
        if sorted_val > best + 1e-9:
            sorted_optimal_everywhere = False
            print("  COUNTEREXAMPLE", c, rates, q, sorted_val, best)

    print("  Claim 2: max |direct - via coverage| = %.2e" % worst_id_err)
    print("  Claim 3: sorted assignment attained the optimum in every trial: %s"
          % sorted_optimal_everywhere)

    # omega_i(r) non-increasing in r (Sec. 3.1)
    mono = True
    for _ in range(2000):
        C = random.randint(3, 40)
        m = random.randint(1, 5)
        rates = sorted(random.sample([1.0, 0.5, 0.25, 0.125, 0.0625], m))
        w = [random.random() for _ in range(m)]
        s = sum(w) or 1.0
        q = [x / s for x in w]
        n, _ = coverage(C, rates, q)
        if any(n[i + 1] > n[i] + 1e-12 for i in range(len(n) - 1)):
            mono = False
    print("  n(k) non-increasing in k for every random q: %s" % mono)

    # Corollary, Eq. (5); the zero cases are those of the Remark
    print("\nClaim 4 (gap = C * Cov(1-n, cbar - c_sorted)):")
    for label, c, rates, q in [
        ("spread capacities, mixed rates",
         [3.0, 2.2, 1.5, 0.9, 0.4, 0.1], [1.0, 0.5, 0.25, 0.125], [.25] * 4),
        ("EQUAL capacities (gap must be 0)",
         [1.0] * 6, [1.0, 0.5, 0.25, 0.125], [.25] * 4),
        ("all clients rate 1 -> n == 1 (gap must be 0)",
         [3.0, 2.2, 1.5, 0.9, 0.4, 0.1], [1.0], [1.0]),
        ("uniform capacity at rate 1/4 (n is a step, gap stays large)",
         [3.0, 2.2, 1.5, 0.9, 0.4, 0.1], [0.25], [1.0]),
    ]:
        C = len(c)
        n, ks = coverage(C, rates, q)
        order = sorted(range(C), key=lambda i: -c[i])
        sorted_val = pop_bound(c, tuple(order), ks, q)
        perms = list(itertools.permutations(range(C)))
        avg_val = sum(pop_bound(c, p, ks, q) for p in perms) / len(perms)
        cbar = sum(c) / C
        cs = sorted(c, reverse=True)
        a = [1.0 - x for x in n]
        b = [cbar - cs[k] for k in range(C)]
        abar = sum(a) / C
        cov = sum((a[k] - abar) * b[k] for k in range(C)) / C
        print("  %-52s gap=%.4f  C*Cov=%.4f" % (label, avg_val - sorted_val, C * cov))


if __name__ == '__main__':
    main()
