class WarmupScheduler:

    def __init__(self, optimizer, model_size, max_lr, warmup_steps):
        self.optimizer = optimizer

        self.warmup_steps = warmup_steps
        self.factor = max_lr / self._get_rate(warmup_steps)
        self.model_size = model_size

        self.cur_step = 0
        self.rate = None

    def _get_rate(self, step):
        return self.model_size ** -0.5 * min(step ** -0.5, step * self.warmup_steps ** -1.5)

    def step(self):
        self.cur_step += 1
        self.rate = self.factor * self._get_rate(self.cur_step)

        for p in self.optimizer.param_groups:
            p["lr"] = self.rate

    def get_last_lr(self):
        if self.rate is None:
            raise RuntimeError("The learning rate is not set yet")

        return [self.rate]
