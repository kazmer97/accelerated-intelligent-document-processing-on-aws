# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

import cfnresponse
import time


def handler(event, context):
    lowercase = event["ResourceProperties"].get("InputString", "").lower()
    lowercaseWithTimestamp = f"{lowercase}-{time.time_ns()}"  # make unique
    responseData = {
        "Lowercase": lowercase,
        "OutputString": lowercaseWithTimestamp,  # don't change key - avoid UserPool update errors
    }
    cfnresponse.send(event, context, cfnresponse.SUCCESS, responseData)
