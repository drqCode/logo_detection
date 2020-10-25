from detecto import core, utils, visualize

image = utils.read_image('tests/image1.jpg')
model = core.Model.load('un_logo_model.pth', ['un_logo'])
#model = core.Model()


labels, boxes, scores = model.predict(image)
print(scores)

visualize.show_labeled_image(image, boxes, labels)
