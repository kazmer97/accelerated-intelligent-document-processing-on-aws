# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

import cfnresponse
import logging
import os

# Get logging level from environment variable with INFO as default
log_level = os.environ.get("LOG_LEVEL", "INFO")
logger = logging.getLogger()
logger.setLevel(getattr(logging, log_level))


def handler(event, context):
    logger.info(event)
    urls = event["ResourceProperties"].get("WebCrawlerURLs", "").split(",")
    seedUrls = [{"Url": url.strip()} for url in urls]
    responseData = {"SeedUrls": seedUrls}
    cfnresponse.send(event, context, cfnresponse.SUCCESS, responseData)
