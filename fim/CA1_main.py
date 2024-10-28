import fim_resources as fim
import itertools

import pyfim # for comparison

Itemset = tuple[int] # We represent itemset as a tuple of item IDs sorted by the "<=" ordering
Transaction = frozenset[int]
Rule = tuple[Itemset, Itemset]

def all_subsets(s: Itemset) -> list[Itemset]:

    res = []
    for i in range( len(s) + 1 ):
        # itertools.combinations preserves ordering
        res.extend( itertools.combinations(s, i) )

    return res

# --- Task 1
def support(tracts: list[Transaction], itemset: Itemset) -> int:
    """
    Counts support of <itemset> in <tracts>
    """
    total = 0
    for tract in tracts:
        if tract.issuperset(itemset):
            total += 1

    return total
# ---

# --- Task 2
def generate_next(F: list[Itemset], n_items: int, k: int) -> list[Itemset]:
    """
    Generates new itemsets from itemsets in <F> of length <k> with limit <n_items>

    assuming ordering of items by ID
    """

    C = []
    for itemset in F:
        assert len(itemset) == k - 1

        start = 0 if len(itemset) == 0 else max(itemset) + 1
        for i in range(start, n_items):
            C.append( itemset + (i,) )

    return C

def filter_violating(frequent: list[Itemset], C: list[Itemset]) -> bool:
    """
    Filters itemsets from <C> which violate downward closure property.
    <frequent> contains frequent itemsets of smaller length.
    """
    F = []
    for itemset in C:
        success = True
        for subs in all_subsets(itemset):

            if len(subs) == len(itemset):
                continue

            if subs not in frequent:
                success = False
                break

        if success:
            F.append(itemset)

    return F

def apriori(tracts: list[Transaction], n_items: int, threshold: int) -> dict[Itemset, int]:
    """
    Mines frequent items w.r.t <tracts> and minimum support threshold <threshold>.
    <n_items> marks the number of items.
    """

    if len(tracts) < threshold:
        return []

    freq_map = { tuple() : len(tracts) }

    k = 1
    F = [tuple()]

    while k < n_items and len(F) != 0:

        C = generate_next(F, n_items, k)
        F_almost = filter_violating(freq_map.keys(), C)
        supps = [support(tracts, itemset) for itemset in F_almost]

        F = []
        for i in range(len(F_almost)):
            if supps[i] >= threshold:
                freq_map[ F_almost[i] ] = supps[i]
                F.append( F_almost[i] )
        k += 1

    return freq_map
# ---

# --- Task 4
def association_rules(freq_map: dict[Itemset, int],
                      conf_thr: float) -> tuple[ list[Rule], float ]:

    rules = []
    for itemset in freq_map.keys():
        for ante in all_subsets(itemset):

            if len(ante) == 0:
                continue

            cons = tuple(item for item in itemset if item not in ante)
            if len(cons) == 0:
                continue

            c = freq_map[itemset] / freq_map[ante]

            if c >= conf_thr:
                rules.append( ( (ante, cons), c ) )

    return rules
# ---

# --- Task 3
def str_itemset(itemset: Itemset, labels: list[str]) -> str:
    return "{ " + " ".join([labels[item] for item in itemset]) + " }"

def str_frequent(freq_map: dict[Itemset, int], labels: list[str], thr: int) -> str:
    sfreqs = sorted( freq_map.items(), key=lambda x: x[1], reverse=True )[:11]
    print(f"Number of frequent (threshold {thr}): ", len(freq_map))
    print("Frequent itemsets:")
    return [ str_itemset(itemset, labels) + " - " + str(count) for itemset, count in sfreqs ]

def print_rules(rules: list[ tuple[Rule, float] ], labels: list[str], conf_thr: float) -> None:
    print(f"Number of rules (threshold {conf_thr}): ", len(rules))
    print("Rules:")
    for (ante, cons), c in sorted(rules, key=lambda x: x[1], reverse=True)[:11]:
        print( f"{str_itemset(ante, labels)} -> {str_itemset(cons, labels)} (conf={c:.2f})" )

def pizzas_analysis() -> None:
    """
    Example from slides --> test purposes
    """

    tracts, labels = fim.load_matrix(fim.DATASETS["pizzas"]["in_file"])
    print("Number of transactions: ", len(tracts))
    thr = 289
    tic = fim.datetime.datetime.now()
    freq_map = apriori(tracts, len(labels), thr)
    tac = fim.datetime.datetime.now()

    print("---")
    print("Function running time: %s" % (tac-tic))
    print( str_frequent(freq_map, labels, thr) )
    print("---")

    thr = 350
    tic = fim.datetime.datetime.now()
    freq_map = apriori(tracts, len(labels), thr)
    tac = fim.datetime.datetime.now()

    print("---")
    print("Function running time: %s" % (tac-tic))
    print( str_frequent(freq_map, labels, thr) )
    print("---")


    conf_thr = 0.6
    tic = fim.datetime.datetime.now()
    rules = association_rules(freq_map, conf_thr)
    tac = fim.datetime.datetime.now()
    print("Function running time: %s" % (tac-tic))
    print_rules(rules, labels, conf_thr)
    print("---")
    # --

def plants_analysis() -> None:
    """
    Plants dataset
    """
    tracts, labels = fim.load_trans_txt(fim.DATASETS["plants"]["in_file"], contains_ids=True)
    print("Number of transactions: ", len(tracts))

    """
    for thr in [ 2500, 3000, 3500, 4000, 5000 ]:
        tic = fim.datetime.datetime.now()
        freq_map = apriori(tracts, len(labels), thr)
        tac = fim.datetime.datetime.now()

        tic2 = fim.datetime.datetime.now()
        pyfim.eclat(tracts, supp=-thr)
        tac2 = fim.datetime.datetime.now()

        print("---")
        print("Apriori running time: %s" % (tac-tic))
        print("Eclat running time: %s" % (tac2-tic2))
        print( str_frequent(freq_map, labels, thr) )
        print("---")
    """

    freq_map = apriori(tracts, len(labels), 3000)

    for conf_thr in [0.5, 0.6, 0.7, 0.8, 0.9]:
        tic = fim.datetime.datetime.now()
        rules = association_rules(freq_map, conf_thr)
        tac = fim.datetime.datetime.now()
        print("Function running time: %s" % (tac-tic))
        print_rules(rules, labels, conf_thr)
        print("---")

def abalone_analysis() -> None:
    tracts, labels = fim.load_data_txt(fim.DATASETS["abalone"]["in_file"])
    print("Number of transactions: ", len(tracts))

    """
    for thr in [ 400, 500, 600, 700, 800 ]:
        tic = fim.datetime.datetime.now()
        freq_map = apriori(tracts, len(labels), thr)
        tac = fim.datetime.datetime.now()

        tic2 = fim.datetime.datetime.now()
        pyfim.eclat(tracts, supp=-thr)
        tac2 = fim.datetime.datetime.now()

        print("---")
        print("Apriori running time: %s" % (tac-tic))
        print("Eclat running time: %s" % (tac2-tic2))
        print( str_frequent(freq_map, labels, thr) )
        print("---")
    """

    freq_map = apriori(tracts, len(labels), 400)

    for conf_thr in [0.5, 0.6, 0.7, 0.8, 0.9]:
        tic = fim.datetime.datetime.now()
        rules = association_rules(freq_map, conf_thr)
        tac = fim.datetime.datetime.now()
        print("Function running time: %s" % (tac-tic))
        print_rules(rules, labels, conf_thr)
        print("---")

def voting_analysis() -> None:
    tracts, labels = fim.load_data_txt(fim.DATASETS["house"]["in_file"])
    print("Number of transactions: ", len(tracts))

    """
    for thr in [ 30, 40, 50, 60, 70 ]:
        tic = fim.datetime.datetime.now()
        freq_map = apriori(tracts, len(labels), thr)
        tac = fim.datetime.datetime.now()

        tic2 = fim.datetime.datetime.now()
        pyfim.eclat(tracts, supp=-thr)
        tac2 = fim.datetime.datetime.now()

        print("---")
        print("Apriori running time: %s" % (tac-tic))
        print("Eclat running time: %s" % (tac2-tic2))
        print( str_frequent(freq_map, labels, thr) )
        print("---")
    """

    freq_map = apriori(tracts, len(labels), 70)

    for conf_thr in [0.5, 0.6, 0.7, 0.8, 0.9]:
        tic = fim.datetime.datetime.now()
        rules = association_rules(freq_map, conf_thr)
        tac = fim.datetime.datetime.now()
        print("Function running time: %s" % (tac-tic))
        print_rules(rules, labels, conf_thr)
        print("---")
# ---

if __name__ == "__main__":

    # (un)comment the respective function call to run the analysis

    pizzas_analysis()
    plants_analysis()
    abalone_analysis()
    voting_analysis()
