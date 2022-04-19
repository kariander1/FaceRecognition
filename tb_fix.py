import sys

import tensorflow as tf
from tensorflow.core.util.event_pb2 import Event
from os import path

def rename_events(input_path, output_path,factor):
    # Make a record writer
    with tf.io.TFRecordWriter(str(output_path)) as writer:
        # Iterate event records
        for rec in tf.data.TFRecordDataset([str(input_path)]):
            # Read event
            ev = Event()
            ev.MergeFromString(rec.numpy())
            # Check if it is a summary
            if ev.summary:
                # Iterate summary values
                for v in ev.summary.value:
                    # Check if the tag should be renamed
                    v.simple_value = v.simple_value * factor

            writer.write(ev.SerializeToString())

def rename_events_dir(input_dir, output_dir):
    input_dir = path(input_dir)
    output_dir = path(output_dir)
    # Make output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    # Iterate event files
    for ev_file in input_dir.glob('**/*.tfevents*'):
        # Make directory for output event file
        out_file = path(output_dir, ev_file.relative_to(input_dir))
        out_file.parent.mkdir(parents=True, exist_ok=True)
        # Write renamed events
        rename_events(ev_file, out_file)

if __name__ == '__main__':
    input_dir = r'/media/hovav/1df0a4b3-8e8b-4fe3-bd7d-499ea780a5a8/home/liorshai/CGMproject/FaceRecognition/runs/mlp_trained_full_with_interp_25_p/._Accuracy__Training Accuracy'
    output_dir = r'/media/hovav/1df0a4b3-8e8b-4fe3-bd7d-499ea780a5a8/home/liorshai/CGMproject/FaceRecognition/runs/mlp_trained_full_with_interp_25_p/._Accuracy__Training Accuracy_new'
    in_f = r'/media/hovav/1df0a4b3-8e8b-4fe3-bd7d-499ea780a5a8/home/liorshai/CGMproject/FaceRecognition/runs/mlp_pretrained_50_p/._Accuracy__Val Accuracy/events.out.tfevents.1649420355.cgm-61.10770.39'
    out_f = r'/media/hovav/1df0a4b3-8e8b-4fe3-bd7d-499ea780a5a8/home/liorshai/CGMproject/FaceRecognition/runs/mlp_pretrained_50_p/._Accuracy__Val Accuracy/events.out.tfevents.1649420355.cgm-61.10770.39.new'
    rename_events(in_f, out_f,0.96494417)
    print('Done')