import logging
import os
import time
import wandb


class WandBLogger:
    def __init__(self, logdir, rank, debug=False, filename=None, summary=True, step=None, name=None):
        self.logger = None
        self.rank = rank
        self.step = step
        self.logdir_results = os.path.join("logs", "results")
        self.use_logger = summary and rank == 0

        self.debug_flag = debug
        logging.basicConfig(filename=filename, level=logging.INFO, format=f'%(levelname)s:{rank}: %(message)s', force=True)

        if rank == 0:
            logging.info(f"[!] starting logging with name {logdir}")
            self.logger = wandb.init(project="WILSON", name=name)
            if self.debug_flag:
                logging.info(f"[!] Entering DEBUG mode")
            self.state = {}
            self.state_int = {}

    def _transform_tag(self, tag):
        tag = tag + f"/{self.step}" if self.step is not None else tag
        return tag

    def is_not_none(self):
        return self.use_logger

    def close(self):
        if self.is_not_none():
            wandb.finish()

    # Functions for WandB
    def _log(self, tag, value, step=None, intermediate=False):
        if self.is_not_none():
            if intermediate:
                if "iteration" not in self.state_int and step is not None:
                    self.state_int["iteration"] = step
                self.state_int[self._transform_tag(tag)] = value
            else:
                if "epoch" not in self.state and step is not None:
                    self.state["epoch"] = step
                self.state[self._transform_tag(tag)] = value

    def commit(self, intermediate=False):
        if self.is_not_none():
            if intermediate:
                self.logger.log(self.state_int)
                self.state_int = {}
            else:
                self.logger.log(self.state)
                self.state = {}

    def add_config(self, opts):
        if self.is_not_none():
            wandb.config.update(vars(opts))

    def add_scalar(self, tag, scalar_value, step=None, intermediate=False):
        self._log(tag, scalar_value, step, intermediate)

    def add_image(self, tag, image, step=None, intermediate=False):
        self._log(tag, wandb.Image(image.transpose(1, 2, 0)), step, intermediate)

    def add_figure(self, tag, image, step=None, intermediate=False):
        self._log(tag, image, step, intermediate)

    def add_table(self, tag, tbl, step=None):
        if self.is_not_none():
            tag = self._transform_tag(tag)
            columns = [k for k in tbl.keys()]
            data = [[x for x in tbl.values()]]
            my_table = wandb.Table(columns=columns, data=data)
            self._log(tag, my_table, step, False)

    # def add_results(self, results):
    #     if self.is_not_none():
    #         columns = ["Class"] + [f"{x}" for x in range(len(results.values()[0]))]
    #         data = [[k]+[str(x) for x in v.values()] for k, v in results.items()]
    #         my_table = wandb.Table(columns=columns, data=data)
    #         self._log("Results", my_table, None, False)

    # Functions for console
    def print(self, msg):
        logging.info(msg)

    def info(self, msg):
        if self.rank == 0:
            logging.info(msg)

    def debug(self, msg):
        if self.rank == 0 and self.debug_flag:
            logging.info(msg)

    def error(self, msg):
        logging.error(msg)

    # Functions for files
    def log_results(self, task, name, results):
        if self.rank == 0:
            file_name = f"{task}.csv"
            dir_path = self.logdir_results
            path = f"{self.logdir_results}/{file_name}"
            if not os.path.exists(dir_path):
                os.makedirs(dir_path, exist_ok=True)
            text = [str(round(time.time())), name, str(self.step)]
            for val in results:
                text.append(str(val))
            row = ",".join(text) + "\n"
            with open(path, "a") as file:
                file.write(row)

    def log_aggregates(self, task, name, results):
        if self.rank == 0:
            file_name = f"{task}-agg.csv"
            dir_path = self.logdir_results
            path = f"{self.logdir_results}/{file_name}"
            if not os.path.exists(dir_path):
                os.makedirs(dir_path, exist_ok=True)
            text = [str(round(time.time())), name, str(self.step)]
            for val in results:
                text.append(str(val))
            row = ",".join(text) + "\n"
            with open(path, "a") as file:
                file.write(row)


class Logger:

    def __init__(self, logdir, rank, type='torch', debug=False, filename=None, summary=True, step=None, name=None):
        self.logger = None
        self.type = type
        self.rank = rank
        self.step = step
        self.logdir_results = os.path.join("logs", "results")
        self.summary = summary and rank == 0
        if summary:
            if type == 'tensorboardX':
                import tensorboardX
                self.logger = tensorboardX.SummaryWriter(logdir)
            elif type == "torch":
                from torch.utils.tensorboard import SummaryWriter
                self.logger = SummaryWriter(logdir)
            else:
                raise NotImplementedError
        else:
            self.type = 'None'

        self.debug_flag = debug
        logging.basicConfig(filename=filename, level=logging.INFO, format=f'%(levelname)s:{rank}: %(message)s', force=True)

        if rank == 0:
            logging.info(f"[!] starting logging at directory {logdir}")
            if self.debug_flag:
                logging.info(f"[!] Entering DEBUG mode")

    def commit(self, intermediate=False):
        pass

    def close(self):
        if self.logger is not None:
            self.logger.close()
        self.info("Closing the Logger.")

    def add_scalar(self, tag, scalar_value, step=None, intermediate=False):
        if self.is_not_none():
            tag = self._transform_tag(tag)
            self.logger.add_scalar(tag, scalar_value, step)

    def add_image(self, tag, image, step=None, intermediate=False):
        if self.is_not_none():
            tag = self._transform_tag(tag)
            self.logger.add_image(tag, image, step)

    def add_figure(self, tag, image, step=None, intermediate=False):
        if self.is_not_none():
            tag = self._transform_tag(tag)
            self.logger.add_figure(tag, image, step)

    def _transform_tag(self, tag):
        tag = tag + f"/{self.step}" if self.step is not None else tag
        return tag

    def is_not_none(self):
        return self.type != "None"

    def add_config(self, opts):
        self.add_table("Opts", vars(opts))

    def add_table(self, tag, tbl, step=None):
        if self.is_not_none():
            tag = self._transform_tag(tag)
            tbl_str = "<table width=\"100%\"> "
            tbl_str += "<tr> \
                     <th>Term</th> \
                     <th>Value</th> \
                     </tr>"
            for k, v in tbl.items():
                tbl_str += "<tr> \
                           <td>%s</td> \
                           <td>%s</td> \
                           </tr>" % (k, v)

            tbl_str += "</table>"
            self.logger.add_text(tag, tbl_str, step)

    def print(self, msg):
        logging.info(msg)

    def info(self, msg):
        if self.rank == 0:
            logging.info(msg)

    def debug(self, msg):
        if self.rank == 0 and self.debug_flag:
            logging.info(msg)

    def error(self, msg):
        logging.error(msg)

    def log_results(self, task, name, results):
        if self.rank == 0:
            file_name = f"{task}.csv"
            dir_path = self.logdir_results
            path = f"{self.logdir_results}/{file_name}"
            if not os.path.exists(dir_path):
                os.makedirs(dir_path, exist_ok=True)
            text = [str(round(time.time())), name, str(self.step)]
            for val in results:
                text.append(str(val))
            row = ",".join(text) + "\n"
            with open(path, "a") as file:
                file.write(row)

    def log_aggregates(self, task, name, results):
        if self.rank == 0:
            file_name = f"{task}-agg.csv"
            dir_path = self.logdir_results
            path = f"{self.logdir_results}/{file_name}"
            if not os.path.exists(dir_path):
                os.makedirs(dir_path, exist_ok=True)
            text = [str(round(time.time())), name, str(self.step)]
            for val in results:
                text.append(str(val))
            row = ",".join(text) + "\n"
            with open(path, "a") as file:
                file.write(row)