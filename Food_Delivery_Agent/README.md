Overview

I developed a voice-enabled Food Delivery Assistant using the LiveKit Agents framework. The assistant supports natural voice conversations through a complete speech pipeline consisting of Speech-to-Text (STT), a Large Language Model (LLM), and Text-to-Speech (TTS).

The assistant can perform several food-ordering tasks through function calling, including checking order status, displaying the restaurant menu, placing new orders, modifying existing orders, showing the current order, completing checkout, and ending the conversation.

Interruption (Barge-in) Handling

The assistant supports interruption handling, allowing users to interrupt the agent while it is speaking. Adaptive interruption detection is enabled to reduce false interruptions caused by background noise and improve the overall conversation experience.

Adding New Tools Safely

The project follows a modular design where each capability is implemented as an independent function tool. New tools can be added by creating a new function, validating user inputs, handling possible errors, and registering the tool with the agent. This design keeps the assistant maintainable, scalable, and easy to extend without affecting existing functionality.