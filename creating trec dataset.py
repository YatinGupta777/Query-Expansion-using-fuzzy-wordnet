#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 20:26:13 2019

@author: yatingupta
"""
f = open('expanded_queries_NoModel.txt','r')
g = open("Q0","w+")
q = 1
for i in f:
    g.write("<top>\n")
    g.write("<num>"+ str(q) + "</num><title>\n")
    q = q + 1
    g.write(i)
    g.write("</title>\n")
    g.write("</top>\n")

f.close()
g.close()