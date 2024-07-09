import numpy as np
import numpy.typing as npt
from collections import namedtuple

NDArrayInt = npt.NDArray[np.int_]
MFQScore = namedtuple("MFQScore", ["initial_mfq", "mfq_without_network", "mfq_with_network"])


def mfq_without_network(
    current_mfq: int,
    num_episode: int,
) -> NDArrayInt:
    """
    Generate MFQ score without network.
    :param current_mfq:
    :param :
    :return:
    """
    mfq = np.zeros(num_episode + 1, dtype=int)
    mfq[0] = current_mfq
    for t in range(1, num_episode + 1):
        mfq[t] = round(mfq[0] + 0.32 * mfq[t - 1] + np.random.normal(0, 1))
        if mfq[t] >= 66:
            mfq[t] = 66
        if mfq[t] < 0:
            mfq[t] = 0
    return mfq


def mfq_with_network(
    initial_mfq: int,
    num_episodes: int,
    family_support: int,
    friend_support: int,
    increase_support_level: float,
) -> NDArrayInt:
    mfq = np.zeros(num_episodes + 1, dtype=int)
    mfq[0] = initial_mfq
    for t in range(1, num_episodes + 1):
        if mfq[t - 1] >= 28:
            mfq[t] = round(
                mfq[0] + 0.32 * mfq[t - 1] - (0.2 + increase_support_level) * family_support - (
                    0.34 + increase_support_level) * friend_support + np.random.normal(0, 1)
            )
        else:
            mfq[t] = round(mfq[0] + 0.32 * mfq[t - 1] + np.random.normal(0, 1))
        if mfq[t] > 66:
            mfq[t] = 66
        if mfq[t] < 0:
            mfq[t] = 0
    return mfq


def generate_mfq(
    network_id: int,
    mfq_min: int,
    mfq_max: int,
    friend_support: int,
    family_support: int,
    num_episodes: int,
    increase_support_level: float,
) -> MFQScore:
    rng = np.random.default_rng(seed=network_id)
    initial_mfq = rng.integers(mfq_min, mfq_max + 1)
    return MFQScore(
        initial_mfq=initial_mfq,
        mfq_without_network=mfq_without_network(initial_mfq, num_episodes),
        mfq_with_network=mfq_with_network(
            initial_mfq,
            num_episodes,
            family_support,
            friend_support,
            increase_support_level
        ),
    )
