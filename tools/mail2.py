#! /usr/local/bin/python

def send_mail(experiment_name,text):
    SMTPserver = 'smtp.seznam.cz'
    sender =     'd.kubesa@seznam.cz'
    destination = ['d.kubesa@gmail.com']

    USERNAME = "d.kubesa@seznam.cz"
    PASSWORD = "staraemailovaadresa"

    # typical values for text_subtype are plain, html, xml
    text_subtype = 'plain'


    # content="""\
    # Sent from Python
    # """
    content = text

    subject=experiment_name + " finished"

    import sys
    import os
    import re

    from smtplib import SMTP_SSL as SMTP       # this invokes the secure SMTP protocol (port 465, uses SSL)
    # from smtplib import SMTP                  # use this for standard SMTP protocol   (port 25, no encryption)

    # old version
    # from email.MIMEText import MIMEText
    from email.mime.text import MIMEText

    try:
        msg = MIMEText(content, text_subtype)
        msg['Subject']=       subject
        msg['From']   = sender # some SMTP servers will do this automatically, not all

        conn = SMTP(SMTPserver)
        conn.set_debuglevel(False)
        conn.login(USERNAME, PASSWORD)
        try:
            conn.sendmail(sender, destination, msg.as_string())
        finally:
            conn.quit()
    except Exception as exc:
        sys.exit( "mail failed; %s" % str(exc) ) # give a error message
