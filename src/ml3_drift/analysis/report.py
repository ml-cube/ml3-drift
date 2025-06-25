class Report:
    def __init__(
        self, concepts: list[tuple[int, int]], same_distributions: dict[int, list[int]]
    ):
        self.concepts = concepts
        self.same_distributions = same_distributions
