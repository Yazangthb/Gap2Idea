"""Run the frozen repeatability calls in their predetermined sequence."""
import five_paper_repeatability as experiment

for input_name, index in experiment.ORDER:
    experiment.run(input_name, index)
