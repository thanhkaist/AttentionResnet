#this is run script
python main.py --model resnet50 --test
python main.py --model bam_resnet50 --attention channel --test
python main.py --model bam_resnet50 --attention spatial --test
python main.py --model bam_resnet50 --attention joint --test
python main.py --model se_resnet50 --attention channel --test
python main.py --model cbam_resnet50 --attention joint --test
python main.py --model cbam_resnet50 --attention channel --test
python main.py --model cbam_resnet50 --attention spatial --test
python main.py --model resnet34 --test
python main.py --model bam_resnet34 --attention channel --test
python main.py --model bam_resnet34 --attention spatial --test
python main.py --model bam_resnet34 --attention joint --test
python main.py --model se_resnet34 --attention channel --test
python main.py --model cbam_resnet34 --attention joint --test
python main.py --model cbam_resnet34 --attention channel --test
python main.py --model cbam_resnet34 --attention spatial --test
