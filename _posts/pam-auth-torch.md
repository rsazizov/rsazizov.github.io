# Extending linux authentication with PyTorch

Some time ago I learned about PAM, while trying to fix a broken auth system on a linux machine (still failed). However, I learned a lot about how authentication works on modern unix systems (it turns out you can learn many useful things by breaking things).

Recently I got a fun idea for a project, to implement a custom authentication module that uses a deep learning model under the hood. I will take a primitive case of classifying some object on a webcam image, but, in theory, you can do many other things like face verification or voice recognition.

Doing root things is very daunting, so no one should do them before having their morning coffee. That's what we are going to address with this project. Our module will continuously grab images from the webcam and run them through a classification net, until a coffee mug is found! Then, and only then a user can be authenticated.

To start off, we will talk about PAM: how it works and how it's configured. Then, we will look at how to implement a basic PAM module. Finally, we will export EfficientNet model using TorchScript and use it in our module using neat PyTorch C++ interface. Let's start!

## Exploring PAM

In the early days of linux, programs that required authentication (e.g. login, sudo) implemented custom logic to ask a user for password and check it against a hash in `/etc/passwd`. That worked for some time, however there quickly rose a need for a more flexible solution, as alternative auth methods were becoming available. For instance, my laptop comes with a fingerprint reader as well as an iris scanner. Moreover, some users might want to design a custom auth solution based on their security needs (e.g. smart badges, encrypted USB keys, etc...).

As a developer, you cannot possibly include all these auth methods in your program, so PAM was created to address this issue. PAM stands for Pluggable Authentication Modules. It's just that, it abstracts away the authentication scheme behind an API that developers can use. Then, individual auth modules that perform the actual authentication (e.g. prompting for password, verifying fingerprint) can be loaded dynamically.

This gives system administrators the freedom to choose any authentication scheme they want by modifying a simple config file. No need to change any of the programs that require authentication.

Programs that use PAM are called PAM-aware applications. All the programs that require authentication on linux are PAM-aware. Each such application gets its own configuration file in `/etc/pam.d/`. Here is an example of my `/etc/pam.d/`.

```
❯ ls /etc/pam.d/
chage                   groupmod   su-l
chfn                    login      system-auth
chgpasswd               newusers   systemd-user
chpasswd                other      system-local-login
chsh                    passwd     system-login
gdm-autologin           polkit-1   system-remote-login
gdm-fingerprint         runuser    system-services
gdm-launch-environment  runuser-l  usermod
gdm-password            shadow     useradd
gdm-smartcard           sshd       userdel
groupadd                su         
groupdel                sudo       
groupmems               vlock
```

A config file will usually have a list of stacked PAM modules with associated rules. Let's take a look at one of these, say `sudo`. This is what I get on arch linux (these files change from distro to distro):

```
#%PAM-1.0

auth       required                    pam_faillock.so      preauth
# Optionally use requisite above if you do not want to prompt for the password
# on locked accounts.
-auth      [success= 2 default=ignore]  pam_systemd_home.so
auth       [success=1 default=bad]     pam_unix.so          try_first_pass nullok
auth       [default=die]               pam_faillock.so      authfail
auth       optional                    pam_permit.so
auth       required                    pam_env.so
auth       required                    pam_faillock.so      authsucc
# If you drop the above call to pam_faillock.so the lock will be done also
# on non-consecutive authentication failures.

-account   [success=1 default=ignore]  pam_systemd_home.so
account    required                    pam_unix.so
account    optional                    pam_permit.so
account    required                    pam_time.so

-password  [success=1 default=ignore]  pam_systemd_home.so
password   required                    pam_unix.so          try_first_pass nullok shadow sha512
password   optional                    pam_permit.so

-session   optional                    pam_systemd_home.so
session    required                    pam_limits.so
session    required                    pam_unix.so
session    optional                    pam_permit.so
```

Looks cryptic, but it's actually very simple. First line is a comment. Then, we have 4 separate blocks, sort to speak.

Each line represents a rule and follows this syntax:

```
type control module-path module-arguments
```

### module-path

Here `module-path` is a path to a shared object that exports PAM functions (we'll learn about them when we get to implement one). On arch linux, these are stored inside `/usr/lib/security`. PAM ships with many common modules that sysadmins can combine to build a robust authentication scheme. Modules can be configurued through `module-arguments`. Additionally, some PAM modules get their own config files like `pam_time.so` (`/etc/security/time.conf`). 

> `pam_time.so` restricts access depending on user, time and terminal (a typical usecase would be something like disabling root login during the nights or holidays for security reasons).

### module-arguments

For instance, on the third line we call `pam_faillock.so` module with `preauth` argument. `pam_faillock.so` is responsible for locking accounts on consequent failed login attempts.  This particular module accepts one of `{preauth|authfail|authsucc}` arguments, depending on where it is called in the stack. `preauth` checks if the account is locked. `authfail` is used to signal a failed login attempt, which increments some internal counter. `authsucc` is called on a successful authentication to reset the internal counter.

### control

Control indicates priority of a module. Most common options are:

* required
  
  Required module, as the name suggest, are obligatory to succeed in order to auhtenticate the user. If a required module fails (e.g. incorrect password), the whole authentication is failed. However, the user is not notified until all the required modules are executed. This is done for security reasons.

* sufficient
  
  If such a module returns PAM_SUCCESS (and no required modules were failed this far), the authentication is immediately considered successful, without executing other modules.

* optional
  
  An optional module is only considered when it's the only module in the stack. Otherwise, it's result is ignored.

For example:

```
auth       required                    pam_faillock.so      preauth
```

In fact, PAM can do much more than simple authentication. A PAM module can expose 4 interfaces that can be called by a PAM-aware application,

* Auth
  
  This is the part of PAM that we have talked about so far, it is called by PAM-aware application when authentication is required.

* Account
  
  It is usually used to check whether a user can be authenticated in the first place. For instance, 

* Password

* Session

Whew! That's a lot of information. However, it's pretty easy, Let's try to use what we learned to explain `auth` part of the `sudo` config:

```
auth       required                    pam_faillock.so      preauth
```

We call `pam_faillock.so` module with `preauth` argument. `pam_faillock.so` is responsible for locking accounts on consequent failed login attempts. This particular module accepts one of `{preauth|authfail|authsucc}` arguments, depending on where it is called in the stack. `preauth` checks if the account is locked (and denies access if it is). `authfail` is used to signal a failed login attempt, which increments some internal counter. `authsucc` is called on a successful authentication to reset the internal counter.

```
-auth      [success=2 default=ignore]  pam_systemd_home.so
```

Modules that start with `-` are ignored if they are not present on the system. So we optionally load `pam_systemd_home.so` which is a module for `systemd-homed` service, that provides user accounts whic are not dependent on system configuration. We can safely ignore it for our purposes.

```
auth       [success=1 default=bad]     pam_unix.so          try_first_pass nullok
```

`pam_unix.so` implements traditional UNIX authentication, based on hashes stored in `/etc/shadow/` 
