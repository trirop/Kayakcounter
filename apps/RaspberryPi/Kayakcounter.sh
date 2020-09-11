source ./Kayakcounter/bin/activate
python Kayakcounter.py \
--modeldir=Kayakcounter_TFlite_model \
--graph=Kayakcounter_edgetpu.tflite \
--labels=object-detection.txt \
--threshold=0.5 \
--edgetpu
