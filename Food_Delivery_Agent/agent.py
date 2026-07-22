import logging
from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    MetricsCollectedEvent,
    RunContext,
    TurnHandlingOptions,
    cli,
    inference,
    metrics,
    room_io,
    text_transforms,
)
from livekit.agents.beta import EndCallTool
from livekit.agents.llm import function_tool
logger = logging.getLogger("basic-agent")
load_dotenv()
ORDERS = {
    "1001": {
        "customer": "Ahmed",
        "restaurant": "McDonald's",
        "status": "Preparing",
        "eta": "20 minutes"
    },
    "1002": {
        "customer": "Sara",
        "restaurant": "KFC",
        "status": "On the way",
        "eta": "10 minutes"
    },
    "1003": {
        "customer": "Omar",
        "restaurant": "Pizza Hut",
        "status": "Delivered",
        "eta": "Delivered"
    }
}
CURRENT_ORDER = {
    "order_id": None,
    "items": {}
}

@function_tool
async def get_order_status(
    context: RunContext,
    order_id: str,
) -> str:
    """
    Check the status of a food delivery order.

    Args:
        order_id: Customer order ID.
    """

    logger.info(f"Checking order {order_id}")

    order = ORDERS.get(order_id)

    if order is None:
        return "Sorry, I couldn't find an order with that ID."

    return (
        f"Order {order_id} from {order['restaurant']} "
        f"is currently {order['status']}. "
        f"Estimated delivery time is {order['eta']}."
    )


class FoodDeliveryAgent(Agent):

    def __init__(self):
        super().__init__(
            instructions="""
You are a Food Delivery Voice Assistant.

Your job is to help customers with:
- Checking order status
- Answering food delivery questions
- Greeting customers politely

Rules:
- Keep answers short.
- Be friendly.
- Speak only in English.
- If the customer asks about an order, always use the get_order_status tool.
- Never make up an order status.
- If the user asks to checkout, confirm the order, complete the purchase, place the final order, or asks for the total price, use the checkout tool.
When the user wants to order food, use the place_order tool with:
- item_name
- quantity

When the user wants to add more of the same item, increase the quantity.

When the user wants to remove an item, use remove_item with the correct quantity.

When the user asks for the total price, receipt, checkout, or to confirm the order, use the checkout tool.
""",
            tools=[
                get_order_status,
                show_menu,
                place_order,
                remove_item,
                show_current_order,
                checkout,
                EndCallTool(),
            ],
        )

    async def on_enter(self):
        self.session.generate_reply(
            instructions="Greet the customer and ask how you can help."
        )

MENU = {
    "Burger": 8,
    "Pizza": 12,
    "Chicken": 10,
    "Fries": 4,
    "Coke": 2
}


@function_tool
async def show_menu(context: RunContext) -> str:
    """
    Returns the restaurant menu.
    """

    menu_text = "Today's menu is:\n"

    for item, price in MENU.items():
        menu_text += f"{item}: {price}\n"

    return menu_text

@function_tool
async def place_order(
    context: RunContext,
    item_name: str,
    quantity: int = 1,
) -> str:
    """
    Place a new order or add items to the current order.

    Args:
        item_name: Name of the menu item.
        quantity: Number of items.
    """

    global CURRENT_ORDER

    menu_item = None
    for item in MENU:
        if item.lower() == item_name.lower():
            menu_item = item
            break

    if menu_item is None:
        return f"Sorry, {item_name} is not available on the menu."

    if CURRENT_ORDER["order_id"] is None:
        CURRENT_ORDER["order_id"] = str(len(ORDERS) + 1001)

    CURRENT_ORDER["items"][menu_item] = (
        CURRENT_ORDER["items"].get(menu_item, 0) + quantity
    )

    ORDERS[CURRENT_ORDER["order_id"]] = {
        "customer": "Guest",
        "restaurant": "Food Delivery",
        "status": "Preparing",
        "eta": "25 minutes"
    }

    summary = "\n".join(
        f"- {name} x{qty}"
        for name, qty in CURRENT_ORDER["items"].items()
    )

    return (
        f"Order ID: {CURRENT_ORDER['order_id']}\n\n"
        f"Current Order:\n{summary}"
    )

@function_tool
async def remove_item(
    context: RunContext,
    item_name: str,
    quantity: int = 1,
) -> str:
    """
    Removes an item or decreases its quantity from the current order.

    Args:
        item_name: Name of the menu item.
        quantity: Quantity to remove.
    """

    global CURRENT_ORDER

    if CURRENT_ORDER["order_id"] is None:
        return "You don't have an active order."

    menu_item = None
    for item in CURRENT_ORDER["items"]:
        if item.lower() == item_name.lower():
            menu_item = item
            break

    if menu_item is None:
        return f"{item_name} is not in your order."

    CURRENT_ORDER["items"][menu_item] -= quantity

    if CURRENT_ORDER["items"][menu_item] <= 0:
        del CURRENT_ORDER["items"][menu_item]

    if len(CURRENT_ORDER["items"]) == 0:
        CURRENT_ORDER["order_id"] = None
        return "Your order is now empty and has been cancelled."

    summary = "\n".join(
        f"- {name} x{qty}"
        for name, qty in CURRENT_ORDER["items"].items()
    )

    return (
        f"I removed {quantity} {menu_item}(s).\n\n"
        f"Current Order:\n{summary}"
    )

@function_tool
async def show_current_order(
    context: RunContext,
) -> str:
    """
    Shows the current active order.
    """

    global CURRENT_ORDER

    if CURRENT_ORDER["order_id"] is None:
        return "You don't have an active order."

    return (
        f"Order ID: {CURRENT_ORDER['order_id']}\n"
        f"Current items: {', '.join(CURRENT_ORDER['items'])}"
    )

@function_tool
async def checkout(
    context: RunContext,
) -> str:
    """
    Shows the final receipt and total price.
    """

    global CURRENT_ORDER

    if CURRENT_ORDER["order_id"] is None:
        return "You don't have an active order."

    total = 0
    receipt = []

    for item, qty in CURRENT_ORDER["items"].items():
        price = MENU[item]
        subtotal = price * qty
        total += subtotal

        receipt.append(
            f"{item:<10} x{qty}   ${subtotal}"
        )

    receipt_text = "\n".join(receipt)

    return (
        "========== RECEIPT ==========\n"
        f"Order ID: {CURRENT_ORDER['order_id']}\n\n"
        f"{receipt_text}\n"
        "-----------------------------\n"
        f"TOTAL: ${total}\n\n"
        "Estimated delivery time: 25 minutes.\n"
        "Thank you for your order!"
    )

server = AgentServer()


@server.rtc_session()
async def entrypoint(ctx: JobContext) -> None:
    # each log entry will include these fields
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }
    session: AgentSession = AgentSession(
        stt=inference.STT("deepgram/nova-3", language="multi"),
        llm=inference.LLM("openai/gpt-4.1-mini"),
        tts=inference.TTS("cartesia/sonic-3", voice="9626c31c-bec5-4cca-baa8-f8ba9e84c8bc"),
        turn_handling=TurnHandlingOptions(
            interruption={
                "resume_false_interruption": True,
                "false_interruption_timeout": 1.0,
            },
            preemptive_generation={"enabled": True, "max_retries": 3},
        ),
        # blocks interruptions for a few seconds after the agent starts speaking to allow client to calibrate AEC
        aec_warmup_duration=3.0,
        tts_text_transforms=[
            "filter_emoji",
            "filter_markdown",
            text_transforms.replace({"LiveKit": "<<ˈ|l|aɪ|v|k|ɪ|t>>"}),
        ],
        stt_context_options={
            "keyterms": ["LiveKit"],
            "keyterm_detection": {
                "enabled": True,
                "turn_interval": 1, 
            },
        },
    )

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent) -> None:
        if ev.metrics.type == "stt_metrics":
            return
        metrics.log_metrics(ev.metrics)

    async def log_usage():
        logger.info(f"Usage: {session.usage}")

    ctx.add_shutdown_callback(log_usage)

    await session.start(
        agent=FoodDeliveryAgent(),
        room=ctx.room,
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(
            ),
        ),
    )


if __name__ == "__main__":
    cli.run_app(server)