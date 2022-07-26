<!-- Start Content -->
<div class="login-box">

  <div class="login-logo">
    <a>SISNOKEN <b>APP</b></a>
    <a>Sistem Informasi Nomor Kendaraan</a>
  </div>
  <!-- /.login-logo -->
  <div class="login-box-body">
    <p class="login-box-msg">Masuk ke Aplikasi</p>

    <?php $attributes = array("name" => "loginform", "role" => "form");
    echo form_open("auth/login", $attributes); ?>
    <div class="form-group has-feedback <?php echo form_error('email') ? 'has-error' : '' ?>">
      <input class="form-control" name="email" placeholder="Username">
      <span class="glyphicon glyphicon-envelope form-control-feedback"></span>
      <span class="help-block"><?php echo form_error('email'); ?></span>
    </div>
    <div class="form-group has-feedback <?php echo form_error('password') ? 'has-error' : '' ?>">
      <input type="password" class="form-control" name="password" placeholder="Password">
      <span class="glyphicon glyphicon-lock form-control-feedback"></span>
      <span class="help-block"><?php echo form_error('password'); ?></span>
    </div>
    <div class="row">
      <div class="col-xs-4">
        <button type="submit" class="btn btn-primary btn-block btn-flat">Masuk</button>
      </div>
      <!-- /.col -->
    </div>
    <?php echo form_close(); ?>
  </div>
  <!-- /.login-box-body -->
</div>
<!-- /.login-box -->