from agents.coordinator_agent import CoordinatorAgent


class ChildProtectionService:
    def __init__(self, project_root):
        self.coordinator = CoordinatorAgent(project_root)

    def get_child_case_recommendations(self, user_input: str):
        return self.coordinator.invoke_child_cases(user_input)

