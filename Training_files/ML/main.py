import Labeling as lb
import training as tr
import artifacts_generator as ag

FOLDERS = ["Dataset", "Sim"]
MODE = "auto"

lb.label(folders=FOLDERS)

_, run_dir = tr.train(mode=MODE)

ag.generate_artifacts(run=run_dir)


