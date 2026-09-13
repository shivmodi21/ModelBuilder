from fastapi import HTTPException


def api_error(
    status_code: int,
    code: str,
    title: str,
    message: str,
    details=None,
):
    raise HTTPException(
        status_code=status_code,
        detail={
            "success": False,
            "error": {
                "code": code,
                "title": title,
                "message": message,
                "details": details or [],
            },
        },
    )