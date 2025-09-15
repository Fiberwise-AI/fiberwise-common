class Agent:
    """Base agent class for Fiberwise applications.
    
    This class provides the foundation for creating agents that can perform
    automated tasks within the Fiberwise ecosystem.
    
    Attributes:
        id (str): Unique identifier for the agent
        name (str): Human-readable name of the agent
        is_active (bool): Whether the agent is currently active
    """
    
    def __init__(self, id: str, name: str, is_active: bool = True):
        """Initialize the agent with basic properties.
        
        Args:
            id: Unique identifier for the agent
            name: Human-readable name of the agent
            is_active: Whether the agent should start as active (default True)
        """
        self.id = id
        self.name = name
        self.is_active = is_active
    
    def run(self, input_data: dict) -> dict:
        """Execute the agent's main functionality.
        
        Args:
            input_data: Dictionary containing input parameters for the agent
            
        Returns:
            Dictionary containing the results of the agent's execution
            
        Raises:
            RuntimeError: If the agent is not active when run is called
        """
        if not self.is_active:
            raise RuntimeError("Cannot run inactive agent")
            
        return {
            "status": "success",
            "input": input_data,
            "agent_id": self.id
        }
