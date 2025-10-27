# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

import boto3
import cfnresponse
import json
import logging
import os

logger = logging.getLogger()
logger.setLevel(os.environ.get("LOG_LEVEL", "INFO"))


def handler(event, context):
    logger.info(json.dumps(event))

    try:
        source_bucket = event["ResourceProperties"]["SourceBucket"]
        source_prefix = event["ResourceProperties"]["SourcePrefix"]
        target_bucket = event["ResourceProperties"]["TargetBucket"]
        target_prefix = event["ResourceProperties"].get("TargetPrefix", "")

        file_list = event["ResourceProperties"].get("FileList", [])

        s3_client = boto3.client("s3")

        if event["RequestType"] == "Create" or event["RequestType"] == "Update":
            # Copy files explicitly from the provided list
            copied_count = 0

            for relative_file_path in file_list:
                # Skip empty entries
                if not relative_file_path.strip():
                    continue

                # Construct source key
                source_key = f"{source_prefix}/{relative_file_path}"

                # Construct target key with optional target prefix
                if target_prefix:
                    target_key = f"{target_prefix}/{relative_file_path}"
                else:
                    target_key = relative_file_path

                logger.info(
                    f"Copying {source_bucket}/{source_key} to {target_bucket}/{target_key}"
                )

                try:
                    copy_source = {"Bucket": source_bucket, "Key": source_key}
                    s3_client.copy_object(
                        CopySource=copy_source,
                        Bucket=target_bucket,
                        Key=target_key,
                    )
                    copied_count += 1
                except Exception as copy_error:
                    logger.warning(f"Failed to copy {source_key}: {str(copy_error)}")
                    # Continue with other files instead of failing the entire operation

            logger.info(f"Successfully copied {copied_count} configuration files")
            cfnresponse.send(
                event,
                context,
                cfnresponse.SUCCESS,
                {"CopiedFiles": copied_count},
                reason=f"Successfully copied {copied_count} configuration files",
            )

        elif event["RequestType"] == "Delete":
            # For delete, we don't need to clean up the configuration files
            # as they may be needed by other resources
            logger.info("Delete request - no action needed for configuration files")
            cfnresponse.send(
                event,
                context,
                cfnresponse.SUCCESS,
                {},
                reason="Delete completed - configuration files retained",
            )

    except Exception as e:
        logger.error(f"Error: {str(e)}")
        cfnresponse.send(
            event,
            context,
            cfnresponse.FAILED,
            {},
            reason=f"Error copying configuration files: {str(e)}",
        )
