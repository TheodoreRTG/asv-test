#!/bin/python
import requests
import sys

versionsUrlList = []

idBegin=int(sys.argv[1])
idEnd=int(sys.argv[2])
for id in range(idBegin,idEnd):
    url = "https://ci.linaro.org/view/All/job/ldcg-python-manylinux-tensorflow-nightly/{}/consoleText".format(id)
    result = requests.get(url).text
    if "Finished: SUCCESS" in result:
        result_list = result.split('\n')
        idx = -1
        curidx = 0
        hits = 0
        for item in result_list:
            if "git_log.stdout:" in item:
                hits += 1
                if hits == 1: idx = curidx
            curidx += 1
        commitHash = result_list[idx].split(' ')[3].replace("'",'')
        buildUrl = result_list[-4].split(' ')[-1].replace("'",'')
        v = [commitHash, "{}tensorflow-aarch64/".format(buildUrl)]
        versionsUrlList.append(v)

fullUrl = []
for k in versionsUrlList:
    url = k[1]
    commit = k[0]
    result = requests.get(url).text
    html = result.split(' ')
    for item in html:
        if "cp39-cp39" in item:
            if "href" in item:
                thisUrl = "https://snapshots.linaro.org{}".format(item.replace("href=", "").replace('"','').replace('\n',''))
                thisEntry = [commit, thisUrl]
                fullUrl.append(thisEntry)

[ print(x,y) for x,y in fullUrl]
