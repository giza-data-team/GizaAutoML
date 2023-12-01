from django.db import models


class DatasetTypeEnum(models.TextChoices):
    Uni = "UniVariate"
    Multi = "MultiVariate"
