from agents.coordinator_agent import CoordinatorAgent

class MentalHealthService:
    def __init__(self, project_root):
        self.coordinator = CoordinatorAgent(project_root)

    def get_mental_health(self, user_input: str):
        return self.coordinator.invoke_mental_health(user_input)