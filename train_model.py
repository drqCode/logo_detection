from detecto import core, utils, visualize

dataset = core.Dataset('images/')
model = core.Model(['un_logo'])

model.fit(dataset)

model.save('un_logo_model.pth')
