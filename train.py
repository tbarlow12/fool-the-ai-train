from ftai_utils.image_repos import FTAI_Images, KaggleImages, ImageNetImages
from ftai_utils import files_in_dir
import shutil
import os
from os import listdir
from azure.cognitiveservices.vision.customvision.training import training_api
from azure.cognitiveservices.vision.customvision.training.models import ImageUrlCreateEntry
import time

PROJECT_NAME = 'Fool the AI'
PROJECT_DESCRIPTION = 'Gamification of image training data collection and model refinement'
PROJECT_DOMAIN_NAME = 'General (compact)'

ftai_dir = 'ftai_images'

def prepare_local():
    # Clear out local directory
    if not os.path.exists(ftai_dir):
        os.makedirs(ftai_dir)
    shutil.rmtree(ftai_dir)
    ftai = FTAI_Images()
    # Downloads images from a container to a local directory
    # For ftai, it defaults to the 'approved' container. You shouldn't
    # need to worry about any other containers. Both kaggle and imagenet
    # are their own containers
    ftai.download_images(ftai_dir)
    return ftai

def get_trainer():
    training_key = os.environ['TRAINING_KEY']
    return training_api.TrainingApi(training_key)

def get_domain_by_name(trainer, name):
    domains = trainer.get_domains()
    for d in domains:
        if d.name == name:
            return d

def get_or_create_project(trainer, name):
    projects = trainer.get_projects()
    for p in projects:
        if p.name == name:
            return p
    return trainer.create_project(
        PROJECT_NAME, 
        description=PROJECT_DESCRIPTION, 
        domain_id=get_domain_by_name(trainer, PROJECT_DOMAIN_NAME).id)

def get_or_create_tags(trainer, project):
    existing_tags = trainer.get_tags(project.id)
    tags = {}
    for t in existing_tags:
        tags[t.name] = t
    for filename in files_in_dir(ftai_dir):
        name = filename.split('---')[0]
        if name in tags:
            tag = tags[name]
        else:
            tag = trainer.create_tag(project.id, name)
            tags[name] = tag
    return tags

def upload_tagged_images(trainer, project, tags):
    for image in os.listdir(os.fsencode(ftai_dir)):
        tag = tags[image.split(b'---')[0].decode('utf-8')]
        with open(ftai_dir + '/' + os.fsdecode(image), mode='rb') as img_data:
            trainer.create_images_from_data(project.id, img_data, [tag.id])
            print('Uploaded', image)

def train_model(trainer, project):
    print('Training...')
    iteration = trainer.train_project(project.id)
    while (iteration.status != "Completed"):
        iteration = trainer.get_iteration(project.id, iteration.id)
        print ("Training status: " + iteration.status)
        time.sleep(1)
    # The iteration is now trained. Make it the default project endpoint
    trainer.update_iteration(project.id, iteration.id, is_default=True)
    print ("Done!")

def main():
    ftai_repo = prepare_local()
    trainer = get_trainer()
    project = get_or_create_project(trainer, 'Fool the AI')
    tags = get_or_create_tags(trainer, project)
    upload_tagged_images(trainer, project, tags)
    train_model(trainer, project)
    # TODO uncomment this when ready for real deal: ftai_repo.processed(ftai_dir)

if __name__ == '__main__':
    main()
