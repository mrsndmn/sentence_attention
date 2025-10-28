import argparse

from mls.manager.job.utils import training_job_api_from_profile
from tqdm import tqdm

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--description_filter", type=str, required=True)
    parser.add_argument(
        "--status",
        default="Pending",
        required=True,
        choices=["Completed", "Completing", "Deleted", "Failed", "Pending", "Running", "Stopped", "Succeeded", "Terminated"],
    )
    parser.add_argument("--limit", default=100, type=int)
    parser.add_argument("--offset", default=0, type=int)
    args = parser.parse_args()

    client, extra_options = training_job_api_from_profile("default")

    all_jobs = client.get_list_jobs(
        region=extra_options["region"],
        status=args.status,
        allocation_name="alloc-officecds-multimodal-2-sr004",
        limit=args.limit,
        offset=args.offset,
    )

    list_jobs_to_delete = []

    for job in all_jobs["jobs"]:
        if args.description_filter in job["job_desc"]:
            list_jobs_to_delete.append(job["job_name"])

            print("Job will be deleted: ", job["job_name"], job["job_desc"])

    if len(list_jobs_to_delete) == 0:
        print("No jobs to delete")
        exit()

    confirm = input("Are you sure you want to delete these jobs? (y/n): ")
    if confirm != "y":
        print("Jobs will not be deleted")
        exit()

    for job_name in tqdm(list_jobs_to_delete, desc="Deleting jobs"):
        resp = client.delete_job(name=job_name, region=extra_options["region"])
        if resp["status"] != "deleted":
            print("Job deletion failed: ", job_name, resp)
