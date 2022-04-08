from timeit import default_timer as timer

import wandb


class WandbWriter:

    def __init__(self, project_name, config):
        self.writer = None
        self.selected_module = ""

        wandb.login()

        wandb.init(project=project_name, config=config)

        self.step = 0
        self.mode = None
        self.timer = timer()

    def set_step(self, step, mode):
        step_skip = step - self.step
        self.step = step
        self.mode = mode
        if step > 0:
            self.add_scalar("steps_per_sec", step_skip / (timer() - self.timer))

        self.timer = timer()

    def get_logging_name(self, name):
        return "_".join((name, self.mode))

    def add_scalar(self, name, value):
        wandb.log({self.get_logging_name(name): value}, step=self.step)

    def add_text(self, name, text):
        wandb.log({self.get_logging_name(name): wandb.Html(text)}, step=self.step)
