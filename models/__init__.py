from models.resnet import *

def get_model(name, norm, attention, num_classes=100):
    if name == 'resnet50':
        return resnet50(attention=attention, norm=norm, num_classes=num_classes)
    if name == 'resnet':
        return resnet50(attention=attention, norm=norm, num_classes=num_classes)
    if name == 'se_resnet50':
        return se_resnet50(attention=attention, norm=norm, num_classes=num_classes)
    if name == 'bam_resnet50':
        return bam_resnet50(attention=attention, norm=norm, num_classes=num_classes)
    if name == 'cbam_resnet50':
        return cbam_resnet50(attention=attention, norm=norm, num_classes=num_classes)
    if name == 'resnet34':
        return resnet34(attention=attention, norm=norm, num_classes=num_classes)
    if name == 'se_resnet34':
        return se_resnet34(attention=attention, norm=norm, num_classes=num_classes)
    if name == 'bam_resnet34':
        return bam_resnet34(attention=attention, norm=norm, num_classes=num_classes)
    if name == 'cbam_resnet34':
        return cbam_resnet34(attention=attention, norm=norm, num_classes=num_classes)
    else:
        raise Exception('Unknown model ', name)


