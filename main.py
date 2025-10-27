import os
import sys
import logging
import asyncio
from tuber_orchestrator import TuberOrchestratorAI
from config import Config
from embedding_service import EmbeddingService
from command_handler import CommandHandler

# Configure logging
logging.basicConfig(level=Config.LOG_LEVEL, format="%(asctime)s - %(levelname)s - %(message)s")

async def main():
    # Ensure .env file exists and API keys are set
    if not os.path.exists(".env"):
        logging.warning("No .env file found. Please create one with your LLM_PROVIDER and API_KEY.")
        logging.warning("Example .env content:\nOPENAI_API_KEY=\"your_key_here\"\nLLM_PROVIDER=\"openai\"")
        sys.exit(1)

    try:
        embedding_service = EmbeddingService()
        orchestrator = TuberOrchestratorAI(embedding_service=embedding_service)
    except ValueError as e:
        logging.error(f"Initialization error: {e}. Please check your .env file and configuration.")
        sys.exit(1)

    # Seed the root vision (the core philosophy of your system)
    root_vision = "دردتنا مبنية على Umbrella Architecture كـ Collaborative Ecosystem، تعمل كـ Meta-System ينسج Social Fabric مرنًا بين المشاركين؛ كل ذلك موجهًا من خلال Holistic Framework ومنسقًا عبر Holarchy of Communities of Practice، مما يعزز Collective Intelligence و Peer-to-Peer collaboration."
    orchestrator.seed_root_vision(root_vision)

    print("\n--- TuberOrchestratorAI Initialized ---")
    print("Type \'help\' for available commands or \'exit\' to quit.")
    print("---------------------------------------")

    command_handler = CommandHandler(orchestrator)
    while True:
        try:
            user_input = await asyncio.to_thread(input, "\nDeveloper> ")
            user_input = user_input.strip()
            if not user_input:
                continue

            if await command_handler.handle_command(user_input):
                break

        except (KeyboardInterrupt, EOFError):
            print("\nExiting TuberOrchestratorAI. Goodbye!")
            break
        except Exception as e:
            logging.error(f"An error occurred in the main loop: {e}", exc_info=True)
            print(f"An unexpected error occurred: {e}. Please check the logs.")

if __name__ == "__main__":
    asyncio.run(main())
