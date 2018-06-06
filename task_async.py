from celery import Celery

app = Celery('tasks', broker='redis://localhost:6379/1', backend='redis://localhost:6379/0', include=['classify'])


# celery -A task_async worker --autoscale=4,3 --purge --discard
