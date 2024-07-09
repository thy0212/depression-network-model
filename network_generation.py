import numpy as np
from collections import namedtuple

NetworkStructure = namedtuple(
    "NetworkStructure", [
        "network_id",
        "family_nodes",
        "friend_nodes",
        "family_active_nodes",
        "friend_active_nodes",
        "family_support",
        "friend_support",
    ]
)


def gen_social_network(
    rng,
    family_min: int,
    family_max: int,
    friend_min: int,
    friend_max: int,
    closest_layer_nodes: int = 5,
    p_connection_1: float = 0.36,
    p_connection_2: float = 0.15,
) -> tuple[int, int, int, int]:
    """
    Generate the social network structure.
    :return: (family_nodes, friend_nodes, family_active_nodes, friend_active_nodes)
    """
    # Generate the network structure (sample the number of friend and family regardless of the max_node)
    family_nodes = rng.integers(family_min, family_max + 1)
    friend_nodes = rng.integers(friend_min, friend_max + 1)
    # Assign the probability of connection between agent i and family/friend
    # the nodes within the closest layer will have p_connection_1 and the rest will have p_connection_2
    if (family_nodes + friend_nodes) <= closest_layer_nodes:
        # Binomial distribution describes how many occurrences of 1
        # given the number of trials and probability of 1 occurs.
        family_active_nodes = rng.binomial(family_nodes, p_connection_1)
        friend_active_nodes = rng.binomial(friend_nodes, p_connection_1)
    else:
        # To select n elements from a joint sample of
        # x elements A and y elements B, we can put them in a sample
        # (x of A and y of B) and simply sample without replacement.
        # Then we count how many of them are sampled.
        # A can be 1 and B can be 0. Thus, after sampling,
        # simply summing them up gives us the number of A is sampled.
        family_elems = np.ones(family_nodes)
        friend_elems = np.zeros(friend_nodes)

        family_closest_nodes = int(
            np.sum(
                rng.choice(
                    np.concatenate([family_elems, friend_elems]),
                    size=closest_layer_nodes,
                    replace=False
                )
            )
        )
        friend_closest_nodes = closest_layer_nodes - family_closest_nodes

        family_active_nodes = rng.binomial(family_closest_nodes, p_connection_1) + \
                              rng.binomial(family_nodes - family_closest_nodes, p_connection_2)

        friend_active_nodes = rng.binomial(friend_closest_nodes, p_connection_1) + \
                              rng.binomial(friend_nodes - friend_closest_nodes, p_connection_2)

    return family_nodes, friend_nodes, family_active_nodes, friend_active_nodes


def gen_support_network(
    rng,
    max_nodes: int,
    family_active_nodes: int,
    friend_active_nodes: int,
) -> tuple[int, int]:
    if (family_active_nodes + friend_active_nodes) <= max_nodes:
        family_support = family_active_nodes
        friend_support = friend_active_nodes
    else:
        # If total connection we generated is more than the maximum number of nodes,
        # we need to randomly pick max_nodes from those family and friend connection.
        # In that case, we can reassign: family_connection, friend_connection.
        family_elems = np.ones(family_active_nodes)
        friend_elems = np.zeros(friend_active_nodes)

        family_support = int(
            np.sum(
                rng.choice(
                    np.concatenate([family_elems, friend_elems]),
                    size=max_nodes,
                    replace=False
                )
            )
        )
        friend_support = max_nodes - family_support

    return family_support, friend_support


def gen_single_network(
    network_id: int,
    family_min: int,
    family_max: int,
    friend_min: int,
    friend_max: int,
    max_nodes: int,
    closest_layer_nodes: int,
    p_connection_1: float,
    p_connection_2: float,
) -> NetworkStructure:
    """
    Generate a single network structure with the given parameters.
    :return: NetworkStructure
    """
    rng = np.random.default_rng(seed=network_id)

    (family_nodes, friend_nodes, family_active_nodes, friend_active_nodes) = gen_social_network(
        rng, family_min, family_max, friend_min, friend_max, closest_layer_nodes, p_connection_1, p_connection_2
    )

    (family_support, friend_support) = gen_support_network(
        rng, max_nodes, family_active_nodes, friend_active_nodes
    )
    return NetworkStructure(
        network_id, family_nodes, friend_nodes, family_active_nodes,
        friend_active_nodes, family_support, friend_support
    )
