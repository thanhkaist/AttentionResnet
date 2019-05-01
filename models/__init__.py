from models.resnet import *

def get_model(name,norm, attention):
    if name == 'resnet50':
        return resnet50(attention=attention, norm=norm)
    if name == 'resnet':
        return resnet50(attention=attention,norm =norm)
    if name == 'se_resnet50':
        return se_resnet50(attention=attention, norm=norm)
    if name == 'bam_resnet50':
        return bam_resnet50(attention=attention, norm=norm)
    if name == 'cbam_resnet50':
        return cbam_resnet50(attention=attention, norm=norm)
    if name == 'resnet34':
        return resnet34(attention=attention, norm=norm)
    if name == 'se_resnet34':
        return se_resnet34(attention=attention, norm=norm)
    if name == 'bam_resnet34':
        return bam_resnet34(attention=attention, norm=norm)
    if name == 'cbam_resnet34':
        return cbam_resnet34(attention=attention, norm=norm)
    else:
        raise Exception('Unknown model ', name)


