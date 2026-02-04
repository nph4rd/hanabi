from dataclasses import dataclass
from typing import Optional

# Valid colors and ranks - configs must be subsets of these
VALID_COLORS = ("R", "Y", "G", "W", "B")
VALID_RANKS = (1, 2, 3, 4, 5)


def _default_card_distribution(ranks: tuple[int, ...]) -> tuple[int, ...]:
    """Generate default card distribution based on ranks.

    Standard Hanabi distribution:
    - Lowest rank: 3 copies per color
    - Middle ranks: 2 copies per color
    - Highest rank: 1 copy per color (irreplaceable!)

    For 5 ranks: (1,1,1,2,2,3,3,4,4,5) = 10 cards per color
    For 3 ranks: (1,1,1,2,2,3) = 6 cards per color
    For 2 ranks: (1,1,1,2) = 4 cards per color
    """
    distribution: list[int] = []
    for i, rank in enumerate(ranks):
        if i == 0:  # Lowest rank: 3 copies
            distribution.extend([rank, rank, rank])
        elif i == len(ranks) - 1:  # Highest rank: 1 copy
            distribution.append(rank)
        else:  # Middle ranks: 2 copies
            distribution.extend([rank, rank])
    return tuple(distribution)


@dataclass(frozen=True)
class GameConfig:
    colors: tuple[str, ...] = VALID_COLORS
    ranks: tuple[int, ...] = VALID_RANKS
    hand_size: int = 5
    card_distribution: Optional[tuple[int, ...]] = None
    max_info_tokens: int = 8
    max_life_tokens: int = 3

    def __post_init__(self) -> None:
        # Validate colors are subset of valid colors
        for color in self.colors:
            if color not in VALID_COLORS:
                raise ValueError(f"Invalid color '{color}'. Must be one of {VALID_COLORS}")

        # Validate ranks
        if len(self.ranks) < 2:
            raise ValueError("Must have at least 2 ranks")
        for rank in self.ranks:
            if rank not in VALID_RANKS:
                raise ValueError(f"Invalid rank {rank}. Must be one of {VALID_RANKS}")

        # Validate hand_size
        if self.hand_size < 1:
            raise ValueError("hand_size must be at least 1")

        # Auto-generate card_distribution if not provided
        if self.card_distribution is None:
            object.__setattr__(self, "card_distribution", _default_card_distribution(self.ranks))

    @property
    def num_colors(self) -> int:
        return len(self.colors)

    @property
    def num_ranks(self) -> int:
        return len(self.ranks)

    @property
    def max_rank(self) -> int:
        return max(self.ranks)

    @property
    def deck_size(self) -> int:
        assert self.card_distribution is not None
        return self.num_colors * len(self.card_distribution)

    @property
    def max_score(self) -> int:
        """Perfect score: all fireworks completed to max rank."""
        return self.num_colors * self.max_rank


# Default config (standard Hanabi: 5 colors, ranks 1-5, hand size 5)
CONFIG = GameConfig()
