import logging
from typing import Dict, Callable, Awaitable

class CommandHandler:
    def __init__(self, orchestrator: 'TuberOrchestratorAI'):
        self.orchestrator = orchestrator
        self.commands: Dict[str, Callable[..., Awaitable[None]]] = {
            'help': self._show_help,
            'exit': self._exit,
            'status': self._get_status,
            'health': self._get_health,
            'list_experiments': self._list_experiments,
            'chat': self._chat,
            'suggest_code': self._suggest_code,
            'propose_experiment': self._propose_experiment,
            'run_experiment': self._run_experiment,
            'validate_code': self._validate_code,
        }

    async def handle_command(self, user_input: str) -> bool:
        """Parses and handles a user command."""
        parts = user_input.strip().split(" ", 1)
        command = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else ""

        if command in self.commands:
            try:
                await self.commands[command](arg)
                return command == 'exit'
            except Exception as e:
                logging.error(f"Error handling command '{command}': {e}", exc_info=True)
                print(f"An error occurred while executing '{command}'. Check logs.")
        else:
            print("Unknown command. Type 'help' for available commands.")
        return False

    async def _show_help(self, _=None):
        print("\nAvailable Commands:")
        print("  exit                                  - Exit the application.")
        print("  help                                  - Show this help message.")
        print("  status                                - Get current system status.")
        print("  chat <message>                        - Send a general message to the AI.")
        print("  suggest_code <problem_desc>           - Ask AI to suggest code for a problem.")
        print("  propose_experiment <goal>             - Ask AI to propose an automated experiment.")
        print("  run_experiment <exp_id>               - Run a proposed experiment.")
        print("  validate_code <suggestion_id>         - Validate a cached code suggestion.")
        print("  list_experiments                      - List all proposed and completed experiments.")
        print("  health                                - Get a system health report.")
        print("\nExample Usage:")
        print("  Developer> chat How can I improve tuber pruning?")
        print("  Developer> suggest_code \"Implement a new reward function\"\n")

    async def _exit(self, _=None):
        print("Exiting TuberOrchestratorAI. Goodbye!")

    async def _get_status(self, _=None):
        status = self.orchestrator.get_system_status()
        print("\n--- System Status ---")
        for key, value in status.items():
            print(f"  {key}: {value}")
        print("---------------------")

    async def _get_health(self, _=None):
        health_report = await self.orchestrator.developer_converse(message="", action_type="analyze_system_health")
        print(health_report)

    async def _list_experiments(self, _=None):
        exp_list = await self.orchestrator.developer_converse(message="", action_type="list_experiments")
        print(exp_list)

    async def _chat(self, arg: str):
        if not arg:
            print("Usage: chat <message>")
            return
        response = await self.orchestrator.developer_converse(message=arg)
        print(f"\nAI Response:\n{response}")

    async def _suggest_code(self, arg: str):
        if not arg:
            print("Usage: suggest_code <problem_description>")
            return
        response = await self.orchestrator.developer_converse(message="", action_type="suggest_code_change", problem_description=arg)
        print(f"\nAI Response:\n{response}")

    async def _propose_experiment(self, arg: str):
        if not arg:
            print("Usage: propose_experiment <goal>")
            return
        response = await self.orchestrator.developer_converse(message="", action_type="propose_automated_experiment", goal=arg)
        print(f"\nAI Response:\n{response}")

    async def _run_experiment(self, arg: str):
        if not arg:
            print("Usage: run_experiment <experiment_id>")
            return
        response = await self.orchestrator.developer_converse(message="", action_type="run_automated_experiment", experiment_id=arg)
        print(f"\nAI Response:\n{response}")

    async def _validate_code(self, arg: str):
        if not arg:
            print("Usage: validate_code <suggestion_id>")
            return
        response = await self.orchestrator.developer_converse(message="", action_type="validate_code_suggestion", suggestion_id=arg)
        print(f"\nAI Response:\n{response}")
