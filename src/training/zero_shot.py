from tqdm import tqdm
import torch
import clip.clip as clip
from .imagenet_zeroshot_data import imagenet_classnames, openai_imagenet_template

import logging
from clip.model import get_feature_inner_product

def zero_shot_classifier(model, classnames, templates, args):
    with torch.no_grad():
        zeroshot_weights = []
        for classname in tqdm(classnames):
            texts = [template(classname) for template in templates] #format with class
            texts = clip.tokenize(texts).to(args.gpu) #tokenize
            if args.distributed:
                class_embeddings, texts_mask = model.module.encode_text(texts)
            elif args.dp:
                class_embeddings, texts_mask = model(None, texts)
            else:
                class_embeddings, texts_mask = model.encode_text(texts)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(args.gpu)
    return zeroshot_weights


def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]

def run(model, classnames, templates, dataloader, args):
    with torch.no_grad():
        top1, top5, n = 0., 0., 0.

        all_embedding = []
        all_masks = []
        for i, classname in enumerate(classnames):
            # texts = [template(classname) for template in templates]  # format with class
            texts = ['a ' + classname]
            texts = clip.tokenize(texts).to(args.gpu)  # tokenize
            class_embeddings, texts_mask = model.module.encode_text(texts)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            all_embedding.append(class_embeddings)
            all_masks.append(texts_mask)

        for images, target in tqdm(dataloader):
            images = images.to(args.gpu)
            target = target.to(args.gpu)

            # predict
            if args.distributed:
                image_features = model.module.encode_image(images)
            elif args.dp:
                image_features = model(images, None)
            else:
                image_features = model.encode_image(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)

            similarity = get_feature_inner_product(image_features, torch.cat(all_embedding, dim=0),
                                                                   torch.cat(all_masks, dim=0))
            # TODO: if more than 1 template is used, we should average the similarity
            logits = 100. * similarity

            # measure accuracy
            acc1, acc5 = accuracy(logits, target, topk=(1, 5))
            top1 += acc1
            top5 += acc5
            n += images.size(0)

    top1 = (top1 / n)
    top5 = (top5 / n)
    return top1, top5

def zero_shot_eval(model, data, epoch, args):

    if 'imagenet-val' not in data and 'imagenet-v2' not in data:
        return {}

    if args.zeroshot_frequency == 0:
        return {}
    if (epoch % args.zeroshot_frequency) != 0 and epoch != args.epochs:
        return {}

    logging.info('Starting zero-shot imagenet.')

    logging.info('Building zero-shot classifier')

    # classifier = zero_shot_classifier(model, imagenet_classnames, openai_imagenet_template, args)

    logging.info('Using classifier')
    results = {}
    if 'imagenet-val' in data:
        top1, top5 = run(model, imagenet_classnames, openai_imagenet_template, data['imagenet-val'].dataloader, args)
        results['imagenet-zeroshot-val-top1'] = top1
        results['imagenet-zeroshot-val-top5'] = top5
    if 'imagenet-v2' in data:
        top1, top5 = run(model, classifier, data['imagenet-v2'].dataloader, args)
        results['imagenetv2-zeroshot-val-top1'] = top1
        results['imagenetv2-zeroshot-val-top5'] = top5


    logging.info('Finished zero-shot imagenet.')

    return results
