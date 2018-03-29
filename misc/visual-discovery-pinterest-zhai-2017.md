
# [Visual Discovery at Pinterest](https://arxiv.org/pdf/1702.04680.pdf)

TLDR: What Pinterest does for visual discovery.

Other papers on enhancing image search with image-related features:
    
    - http://www.kevinjing.com/jing_pami.pdf
        - Google, got image similarity graph, and chose images with high PageRank


## Feature Representation

They choose their image feature representation by taking NN similarity on Pascal dataset for different networks and layers, and seeing which one gives the highest Precision@k. They also binarize the features to make them smaller.

They find that VGG16 fc6 layer with binary features performs well.


## Object Detection

They tried Faster-RCNN and Single shot detection, both are computationally intensive; single shot detection is faster.

## Related images

They use VGG fc6 features, but also add categories to the features. They perform Rank-SVM to show related pins/images to users.

## Object search

They use SSD to extract objects from images offline, and index those objects. They then are able to recommend images online with certain objects in them.
