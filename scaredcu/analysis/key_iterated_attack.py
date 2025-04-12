from scaredcu.selection_functions.iterated import _IteratedAttackSelectionFunctionWrapped
from scaredcu.container import Container
from tqdm.auto import tqdm


class KeyIteratedAttack:

    def __init__(self, selection_function, attack_cls, *attack_args, **attack_kwargs):
        if not isinstance(selection_function, _IteratedAttackSelectionFunctionWrapped):
            raise ValueError("selection_function must be a _IteratedAttackSelectionFunctionWrapped")
        self.selection_function = selection_function
        self.attack_cls = attack_cls
        self.attack_args = attack_args
        self.attack_kwargs = attack_kwargs

    def run(self, ths, step, preprocesses=[], frame=None):

        for _ in tqdm(range(self.selection_function.num_steps)):
            assert self.selection_function.done() == False

            attack = self.attack_cls(*self.attack_args, selection_function=self.selection_function, **self.attack_kwargs)

            for i in range(0, len(ths), step):
                if frame is not None:
                    container = Container(ths[i:i + step], preprocesses=preprocesses, frame=frame)
                else:
                    container = Container(ths[i:i + step], preprocesses=preprocesses)
                attack.run(container)

            if hasattr(attack, 'convergence_traces') and attack.convergence_traces is not None:
                self.selection_function.save_scores(attack.convergence_traces)
            else:
                self.selection_function.save_scores(attack.scores)
            self.selection_function.next()

        return self.selection_function.scores