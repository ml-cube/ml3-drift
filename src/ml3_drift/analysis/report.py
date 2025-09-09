class Report:
    def __init__(
        self, concepts: list[tuple[int, int]], same_distributions: dict[int, list[int]]
    ):
        self.concepts = concepts
        self.same_distributions = same_distributions

    def __repr__(self):
        return f"Report(concepts={self.concepts}, same_distributions={dict(self.same_distributions)})"
