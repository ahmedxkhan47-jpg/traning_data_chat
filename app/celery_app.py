from celery import Celery

celery = Celery(
    "repair_system",
    broker="amqp://guest:guest@localhost:5672/",
    backend="rpc://",   # optional
    
    include=[
        "app.main"
    ]
)

celery.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
)


celery.conf.beat_schedule = {

    "check appointments every minute": {
        "task": "app.main.generate_order",
        "schedule": 120.0,   # seconds
    },

}