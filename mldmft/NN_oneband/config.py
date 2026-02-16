class ConfigDict(dict):
    """
    A dictionary subclass that allows attribute-style access.
    """

    def __init__(self, *args, type_safe=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self
        self._type_safe = type_safe

    def __setitem__(self, key, value):
        if self._type_safe and key in self and not isinstance(value, type(self[key])):
            raise TypeError(f"Cannot change type of key '{key}' from {type(self[key])} to {type(value)}")
        super().__setitem__(key, value)

    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError:
            raise AttributeError(f"'ConfigDict' object has no attribute '{item}'")

    def __setattr__(self, key, value):
        if key in self.__dict__:
            object.__setattr__(self, key, value)
        else:
            self[key] = value

def default() -> ConfigDict:
    return ConfigDict({

        "folders": [],

        "betavals": [],

        "device": None,

        "restart": {
            "iteration": None,
            "filename": None,
        },


        #TODO make a default ansatz parameters
        #"ansatz": {},

        #"loss": {
        #    "ke_kwargs": {},
        #    "pe_kwargs": {},
        #    "mini_batch": None,
        #    "energy_clipping": 5.,
        #    "center_shifting": True,
        #    "clip_from_median": True,
        #},

        "optimize": {
            "iterations": 100,
            "lr": {},
            "optimizer": "Adam",
        },

        "log": {
            "stat_path": "plot_examples/",
            "ckpt_path": "checkpoints/",
            "stat_ckpt_every": 10,

        },

    }, type_safe=False)