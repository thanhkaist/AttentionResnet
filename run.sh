# this is run script
#python main.py --model resnet50 
#python main.py --model bam_resnet50 --attention channel
#python main.py --model bam_resnet50 --attention spatial
#python main.py --model bam_resnet50 --attention joint
#python main.py --model se_resnet50 --attention channel 
python main.py --model cbam_resnet50 --attention joint
python main.py --model cbam_resnet50 --attention channel
python main.py --model cbam_resnet50 --attention spatial
