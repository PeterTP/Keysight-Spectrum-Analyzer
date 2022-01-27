# shop/models.py

from django.db import models


class Data(models.Model):
    frequency = models.FloatField(default=0)
    funit = models.CharField(max_length=255)
    amplitude = models.FloatField(default=0)
    aunit = models.CharField(max_length=255)
    sequence = models.CharField(max_length=255)
    trace = models.IntegerField(default=0)
    r = models.CharField(max_length=255)
    c = models.CharField(max_length=255)
    lab = models.CharField(max_length=255)

    class Meta:
        ordering = ['trace']

    def __str__(self):
        return f'{self.frequency},{self.amplitude},{self.trace},{self.c}'

