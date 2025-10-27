# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

import cfnresponse
import json
import logging
import os

# Initialize logging
logger = logging.getLogger()
logger.setLevel(os.environ.get("LOG_LEVEL", "INFO"))


def handler(event, context):
    logger.info(json.dumps(event))
    input_str = event["ResourceProperties"].get("InputString", "")
    max_length = int(event["ResourceProperties"].get("MaxLength", 0))
    status = cfnresponse.SUCCESS
    reason = f"Stack Name Length under {max_length} - OK"
    if event["RequestType"] == "Create":
        if len(input_str) > max_length:
            status = cfnresponse.FAILED
            reason = f"Stack Name ({input_str}) length ({len(input_str)}) too long - max length {max_length} - FAILED"
    else:
        logger.info(f"Request type is {event['RequestType']} - skipping")
    cfnresponse.send(event, context, status, {}, reason=reason)
