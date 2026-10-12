import Labeling as lb
import training as tr
import artifacts_generator as ag

lb.label()

_, run_dir = tr.train()

ag.generate_artifacts()


