from agents.coordinator_agent import CoordinatorAgent


class ChildProtectionService:
    def __init__(self, project_root):
        self.coordinator = CoordinatorAgent(project_root)

    def get_child_case_recommendations(self, user_input: str):
        return self.coordinator.invoke_child_cases(user_input)

    def get_child_insights(self, district: str):
        return self.coordinator.get_child_cases_insights(district)

    def get_risk_summary(self) -> dict:
        return self.coordinator.get_child_risk_summary()

    # ------------------------------------------------------------------
    # Budget allocation
    # ------------------------------------------------------------------
    def allocate_budget(self,total_budget: float,query: str | None = None,selected_districts: list[str] | None = None) -> dict:

        return self.coordinator.allocate_child_budget(
            total_budget=total_budget,
            query=query,
            selected_districts=selected_districts,
        )