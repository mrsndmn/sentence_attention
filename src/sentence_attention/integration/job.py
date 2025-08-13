import time

from mls.manager.job.utils import training_job_api_from_profile

REGION = "SR004"


def accelerate_config_by_instance_type(instance_type: str, workdir_prefix: str) -> str:

    if instance_type == "a100.8gpu":
        return f"{workdir_prefix}/configs/accelerate/8gpu.yaml"
    elif instance_type == "a100.6gpu":
        return f"{workdir_prefix}/configs/accelerate/6gpu.yaml"
    elif instance_type == "a100.4gpu":
        return f"{workdir_prefix}/configs/accelerate/4gpu.yaml"
    elif instance_type == "a100.2gpu":
        return f"{workdir_prefix}/configs/accelerate/2gpu.yaml"
    elif instance_type == "a100.1gpu":
        return f"{workdir_prefix}/configs/accelerate/1gpu.yaml"
    else:
        raise ValueError(f"Unknown instance type: {instance_type}")


def get_in_progress_jobs(client, statuses=None):
    """
    Example:
        from sentence_attention.integration.job import get_in_progress_jobs
        from mls.manager.job.utils import training_job_api_from_profile

        client, _ = training_job_api_from_profile("default")
        in_progress_jobs = get_in_progress_jobs(client)

    """

    all_in_progress_jobs = []

    if statuses is None:
        statuses = ["Pending", "Running"]

    for non_final_status in statuses:
        while True:
            non_final_jobs = client.get_list_jobs(
                region=REGION,
                allocation_name="alloc-officecds-multimodal-2-sr004",
                status=non_final_status,
                limit=1000,
                offset=0,
            )
            if "jobs" in non_final_jobs:
                break
            elif "error_code" in non_final_jobs and non_final_jobs["error_code"] == [
                32,
                20,
            ]:  # no active session, access_token expired
                print("Error:", non_final_jobs, "try adain")
                time.sleep(5)
                client, _ = training_job_api_from_profile("default")
            else:
                raise ValueError("Unknown error in get_in_progress_jobs:", non_final_jobs)

        all_in_progress_jobs.extend(non_final_jobs["jobs"])

    return all_in_progress_jobs
