import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email.mime.text import MIMEText
from email import encoders
from dotenv import load_dotenv
import os

def send_email(recipient, subject, body, file_path=None):
    smtp_server = "smtp.163.com"
    sender_email = "zhyu_edu@163.com"
    load_dotenv()
    password = os.getenv("email_passwd")
    
    message = MIMEMultipart()
    message["From"] = sender_email
    message["To"] = recipient
    message["Subject"] = subject
    
    message.attach(MIMEText(body, "plain"))
    

    if file_path != None:
        # 添加附件
        with open(file_path, "rb") as attachment:
            part = MIMEBase("application", "octet-stream")
            part.set_payload(attachment.read())
        
        encoders.encode_base64(part)
        
        filename = file_path.split("/")[-1]
        part.add_header(
            "Content-Disposition",
            f"attachment; filename= {filename}",
        )
        
        message.attach(part)
    
    # 发送邮件
    with smtplib.SMTP(smtp_server) as server:
        server.starttls()
        server.login(sender_email, password)
        server.sendmail(sender_email, recipient, message.as_string())
    
    print(f"结果邮件已发送到 {recipient}")


# send_email(
#     "zhyu_edu@163.com",
#     "测试",
#     "这是内容",
#     "v0.2.log"
# )